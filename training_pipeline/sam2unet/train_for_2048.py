import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from torch.cuda.amp import autocast, GradScaler

# 모듈 import
from SAM2UNET import SAM2UNet 
from custom_dataset import XRayDataset, CLASSES

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", default="./checkpoints/sam2_hiera_large.pt", type=str)
parser.add_argument("--train_image_path", default="./data/train/DCM", type=str)
parser.add_argument("--train_mask_path", default="./data/train/outputs_json", type=str)
parser.add_argument('--save_path', default="./sam2_unet_result_checkpoints", type=str)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", default=2, type=int) # 배치 1 필수
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--wandb_project", default="XRay-Bone", type=str)
parser.add_argument("--wandb_name", default="Freeze_Encoder_Training", type=str)
args = parser.parse_args()

def generate_edge_target(target):
    """
    target: [Batch, 29, H, W]
    return: [Batch, 29, H, W] (안쪽으로 3픽셀 두께의 경계선)
    """
    with torch.no_grad():
        # 1. 침식 (Erosion)
        # 3픽셀만큼 깎아내기 위해 kernel_size=7 사용 (반지름=3)
        # padding=3으로 설정하여 이미지 크기 유지
        eroded = -F.max_pool2d(-target, kernel_size=5, stride=1, padding=2)
        
        # 2. 안쪽 경계 추출 (Inner Edge)
        # [원본] - [깎인 것] = [깎여나가는 테두리 부분(안쪽 경계)]
        inner_edge = target - eroded
        
    return inner_edge

def generate_overlap_target(target):
    """
    각 클래스별로 '다른 뼈와 겹치는 영역'만 남기고 나머지는 0으로 만듭니다.
    target: [Batch, 29, H, W]
    return: [Batch, 29, H, W]
    """
    with torch.no_grad():
        # 1. 픽셀별로 뼈가 몇 개 있는지 계산 (Channel-wise Sum)
        # [Batch, 1, H, W]
        count_map = torch.sum(target, dim=1, keepdim=True)
        
        # 2. 뼈가 2개 이상 있는 '분쟁 지역' 찾기 (Binary Mask)
        # 1.0 = 겹침 발생, 0.0 = 겹침 없음(배경 or 뼈 1개)
        global_overlap_mask = (count_map >= 2.0).float()
        
    return global_overlap_mask

def multilabel_loss(pred, target, focal_coef=0.5, dice_coef=0.5):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    gamma = 2.0
    alpha = 0.25
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    focal_loss = focal_loss.mean()
    
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
    dice_loss = 1 - dice_score.mean()
    
    return focal_coef * focal_loss + dice_coef * dice_loss

def compute_dice_score(pred, target, threshold=0.5):
    pred_mask = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_mask * target).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_scores = (2. * intersection + 1e-5) / (union + 1e-5)
    return dice_scores.mean(dim=0)

def validation(model, dataloader, device, num_classes):
    model.eval()
    total_loss = 0
    total_class_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            x = batch['image'].to(device)
            target = batch['label'].to(device).permute(0, 3, 1, 2).float()
            
            with autocast():
                pred0, pred1, pred2 = model(x)

                #overlap_mask = generate_overlap_target(target)
                #edge_mask = generate_edge_target(target)

                loss0 = multilabel_loss(pred0, target)
                loss1 = multilabel_loss(pred1, target)
                loss2 = multilabel_loss(pred2, target)

                #loss_edge = multilabel_loss(edge_pred, edge_mask, focal_coef=1.0, dice_coef=0.0)
                #loss_overlap = multilabel_loss(overlap_pred, overlap_mask, focal_coef=1.0, dice_coef=0.0)

                loss = loss0 + loss1 + loss2 + 0.5
                
            total_loss += loss.item()
            batch_class_dice = compute_dice_score(pred0, target, threshold=0.5)
            total_class_dice += batch_class_dice
            
    avg_loss = total_loss / len(dataloader)
    avg_class_dice = total_class_dice / len(dataloader)
    avg_dice = avg_class_dice.mean().item()
    return avg_loss, avg_dice, avg_class_dice

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    wandb.login(relogin=True, key="userkey")
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

    g = torch.Generator()
    g.manual_seed(1024)

    # Transform: 2048 원본 해상도 사용 (메모리 절약을 위해 Resize 없음)
    train_transform = A.Compose([ToTensorV2()])
    valid_transform = A.Compose([ToTensorV2()])
    
    train_dataset = XRayDataset(args.train_image_path, args.train_mask_path, is_train=True, transforms=train_transform)
    valid_dataset = XRayDataset(args.train_image_path, args.train_mask_path, is_train=False, transforms=valid_transform) 
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    device = torch.device("cuda")
    
    # 모델 로드
    model = SAM2UNet(args.hiera_path)
    model.to(device)

    # Optimizer: requires_grad=True인 파라미터만 학습 (Decoder만 학습)
    optim = opt.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epoch, eta_min=5.0e-4)
    
    scaler = GradScaler() # AMP
    
    best_dice = 0.0
    os.makedirs(args.save_path, exist_ok=True)
    accumulation_steps = 1

    # 메모리 파편화 방지
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()

    print("Start Backbone-Freezed Training...")
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        optim.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epoch} Train")

        for i, batch in pbar:
            x = batch['image'].to(device)
            target = batch['label'].to(device).permute(0, 3, 1, 2).float()

            with autocast():
                # Encoder는 no_grad()로 실행되므로 메모리 거의 안 씀
                pred0, pred1, pred2 = model(x)

                #overlap_mask = generate_overlap_target(target)
                #edge_mask = generate_edge_target(target)
                
                loss0 = multilabel_loss(pred0, target)
                loss1 = multilabel_loss(pred1, target)
                loss2 = multilabel_loss(pred2, target)
                #loss_edge = multilabel_loss(edge_pred, edge_mask, focal_coef=1.0, dice_coef=0.0)
                #loss_overlap = multilabel_loss(overlap_pred, overlap_mask, focal_coef=1.0, dice_coef=0.0)
                
                loss = (loss0 + loss1 + loss2) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            
            current_loss_val = loss.item() * accumulation_steps
            train_loss += current_loss_val

            pbar.set_postfix({'Total': f"{current_loss_val:.4f}", 'Seg': f"{loss0.item():.4f}"})
            wandb.log({
                "Train/Epoch Loss": loss.item(),
                "Train/Mask Loss": loss0.item(),
                #"Train/Edge Loss": loss_edge.item(),
                #"Train/Overlap Loss": loss_overlap.item(),
                "Train/Learning Rate": optim.param_groups[0]['lr']
            })
            # 메모리 즉시 해제 (중요)
            del x, target, pred0, pred1, pred2, loss

        scheduler.step()
        torch.cuda.empty_cache() # 에폭마다 캐시 정리

        val_loss, val_dice, val_class_dice = validation(model, valid_loader, device, 29)
        print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f} | Valid Dice: {val_dice:.4f}")

        log_dict ={
            "Train/Epoch Loss": train_loss/len(train_loader),
            "Valid/Loss": val_loss,
            "Valid/Dice": val_dice,
            "Epoch": epoch + 1
        }

        for i, score in enumerate(val_class_dice):
            class_name = CLASSES[i]
            log_dict[f"Valid_CLASS/{class_name}"]=score.item()
        
        wandb.log({"Valid/Loss": val_loss, "Valid/Dice": val_dice, "Epoch": epoch + 1})

        if val_dice > best_dice:
            print(f"🎉 New Best Model! ({best_dice:.4f} -> {val_dice:.4f})")
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.save_path, 'experiment20.pth'))
            wandb.log({"Best Dice": best_dice})

    wandb.finish()

if __name__ == "__main__":
    seed_torch(1024)
    main(args)