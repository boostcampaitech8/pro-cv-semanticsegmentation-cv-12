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

# --- [User Imports] ---
# 사용자의 파일 구조에 맞춰 import 경로를 유지합니다.
try:
    from SAM2UNet_attention import SAM2AttentionUNet
    from SAM2UNet import SAM2UNet
    from custom_dataset import XRayDataset, CLASSES
except ImportError:
    print("Warning: Custom modules (SAM2UNet, custom_dataset) not found. Please ensure they are in the same directory.")

# --- [Arguments] ---
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", default="./checkpoints/sam2_hiera_large.pt", type=str, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", default="./data/train/DCM", type=str, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", default="./data/train/outputs_json", type=str,
                    help="path to the mask file for training")
parser.add_argument('--save_path', default="./sam2_unet_result_checkpoints", type=str,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", default=100, type=int, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--wandb_project", default="XRay-Bone", type=str, help="wandb project name")
parser.add_argument("--wandb_name", default="experiment", type=str, help="wandb run name")
args = parser.parse_args()

# --- [Loss Function] ---
def multilabel_loss(pred, target, focal_coef=0.5, dice_coef=0.5):
    """
    pred: [Batch, 29, H, W] (Logits)
    target: [Batch, 29, H, W] (0.0 or 1.0)
    """
    # 1. BCE With Logits Loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)

    # Focal Parameter
    gamma = 2.0
    alpha = 0.25 

    # 2. Dice Loss
    pred_sigmoid = torch.sigmoid(pred)

    # Alpha Balancing
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    # Focal Loss Calculation
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    focal_loss = focal_loss.mean()
    
    # Dice Score Calculation
    intersection = (pred_sigmoid * target).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
    dice_loss = 1 - dice_score.mean()
    
    return focal_coef * focal_loss + dice_coef * dice_loss

def compute_dice_score(pred, target, threshold=0.5):
    """
    pred: [B, 29, H, W] (Logits)
    target: [B, 29, H, W] (0.0 or 1.0)
    """
    pred_mask = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_mask * target).sum(dim=(2, 3)) 
    union = pred_mask.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_scores = (2. * intersection + 1e-5) / (union + 1e-5)
    return dice_scores.mean(dim=0)

# --- [Sliding Window Inference Function] ---
def sliding_window_inference(model, image, window_size=1024, overlap=0.5, num_classes=29):
    """
    이미지를 윈도우 단위로 잘라 예측 후 합치는 함수.
    image: [1, C, H, W] (Batch size must be 1 for validation safety)
    """
    b, c, h, w = image.shape
    
    final_pred = torch.zeros((b, num_classes, h, w), device=image.device)
    count_map = torch.zeros((b, num_classes, h, w), device=image.device)
    
    stride = int(window_size * (1 - overlap))
    
    # 1. Padding (윈도우 크기에 맞게 우측/하단 패딩)
    pad_h = 0
    pad_w = 0
    if h < window_size:
        pad_h = window_size - h
    elif (h - window_size) % stride != 0:
        pad_h = stride - ((h - window_size) % stride)
        
    if w < window_size:
        pad_w = window_size - w
    elif (w - window_size) % stride != 0:
        pad_w = stride - ((w - window_size) % stride)
        
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    h_padded, w_padded = image_padded.shape[2], image_padded.shape[3]
    
    # 2. Sliding Window Loop
    # 윈도우가 이미지를 벗어나지 않도록 범위 설정
    for y in range(0, h_padded - window_size + 1, stride):
        for x in range(0, w_padded - window_size + 1, stride):
            patch = image_padded[:, :, y:y+window_size, x:x+window_size]
            
            with torch.no_grad():
                # 모델이 (pred0, pred1, pred2) 튜플을 반환한다고 가정
                # 가장 High Resolution 결과인 첫 번째 요소(pred0)를 사용
                preds = model(patch)
                if isinstance(preds, tuple):
                    pred_patch = preds[0]
                else:
                    pred_patch = preds
            
            # 결과 누적
            final_pred[:, :, y:y+window_size, x:x+window_size] += pred_patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1.0
            
    # 3. Crop & Average
    final_pred = final_pred[:, :, :h, :w]
    count_map = count_map[:, :, :h, :w]
    
    # 겹치는 부분 평균 (나누기 0 방지)
    final_pred = final_pred / torch.clamp(count_map, min=1.0)
    
    return final_pred

# --- [Validation Function] ---
def validation(model, dataloader, device, num_classes):
    model.eval()
    total_loss = 0
    total_class_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating (Sliding Window)"):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            target = target.permute(0, 3, 1, 2).float()
            
            # Batch Size가 1일 때 최적화된 Sliding Window 적용
            if x.size(0) == 1:
                pred0 = sliding_window_inference(
                    model, x, window_size=1024, overlap=0.5, num_classes=num_classes
                )
            else:
                # 만약 배치 사이즈가 1보다 크다면 루프를 돌며 처리 (메모리 보호)
                preds_list = []
                for i in range(x.size(0)):
                    single_pred = sliding_window_inference(
                        model, x[i:i+1], window_size=1024, overlap=0.5, num_classes=num_classes
                    )
                    preds_list.append(single_pred)
                pred0 = torch.cat(preds_list, dim=0)
            
            # Loss 계산 (슬라이딩 윈도우 결과는 Average된 Logit 형태이므로 그대로 Loss 계산 가능)
            loss = multilabel_loss(pred0, target)
            total_loss += loss.item()
            
            # Dice Score 계산
            batch_class_dice = compute_dice_score(pred0, target, threshold=0.5)
            total_class_dice += batch_class_dice
            
    avg_loss = total_loss / len(dataloader)
    avg_class_dice = total_class_dice / len(dataloader)
    avg_dice = avg_class_dice.mean().item()
    return avg_loss, avg_dice, avg_class_dice

# --- [Seeding Functions] ---
def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- [Main Training Loop] ---
def main(args):
    # WandB Init
    wandb.login(relogin=True, key="userkey")
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=args
    )

    # Seed
    seed_torch(1024)
    g = torch.Generator()
    g.manual_seed(1024)

    # Transforms
    train_transform = A.Compose([
        A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0),
        A.CropNonEmptyMaskIfExists(height=1024, width=1024, ignore_values=[0], p=1.0),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        ToTensorV2()
    ])
    
    # Dataset
    train_dataset = XRayDataset(
        image_root=args.train_image_path,
        label_root=args.train_mask_path,
        is_train=True,
        transforms=train_transform
    )
    
    valid_dataset = XRayDataset(
        image_root=args.train_image_path,
        label_root=args.train_mask_path,
        is_train=False,
        transforms=valid_transform
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    # [변경] Validation 시에는 Batch Size 1 권장 (Sliding Window 메모리 이슈)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )

    print(f"Train Size: {len(train_dataset)}, Valid Size: {len(valid_dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = SAM2UNet(args.hiera_path)
    model.to(device)

    # Optimizer & Scheduler
    optim = opt.AdamW([{"params":model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-4)
    
    best_dice = 0.0
    os.makedirs(args.save_path, exist_ok=True)
    
    accumulation_steps = 1 

    print("Start Training...")
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        optim.zero_grad()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epoch} Train")

        for i, batch in pbar:
            x = batch['image'].to(device)
            target = batch['label'].to(device).permute(0, 3, 1, 2).float()

            pred0, pred1, pred2 = model(x)

            loss0 = multilabel_loss(pred0, target)
            loss1 = multilabel_loss(pred1, target)
            loss2 = multilabel_loss(pred2, target)
            
            w0, w1, w2 = 1.0, 1.0, 1.0
            
            loss = (w0*loss0 + w1*loss1 + w2*loss2) / accumulation_steps
            
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
                optim.zero_grad()
            
            current_loss_val = loss.item() * accumulation_steps
            train_loss += current_loss_val

            pbar.set_postfix({
                'Total': f"{current_loss_val:.4f}", 
                'Seg': f"{loss0.item():.4f}", 
            })

            wandb.log({
                "Train/Epoch Loss": current_loss_val,
                "Train/Mask Loss": loss0.item(),
                "Train/Learning Rate": optim.param_groups[0]['lr']
            })

        scheduler.step()

        # Validation (Sliding Window 적용됨)
        val_loss, val_dice, val_class_dice = validation(model, valid_loader, device, 29)

        print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f} | Valid Loss: {val_loss:.4f} | Valid Dice: {val_dice:.4f}")

        log_dict ={
            "Train/Epoch Loss": train_loss/len(train_loader),
            "Valid/Loss": val_loss,
            "Valid/Dice": val_dice,
            "Epoch": epoch + 1
        }

        for i, score in enumerate(val_class_dice):
            # CLASSES 변수가 custom_dataset에 정의되어 있다고 가정
            try:
                class_name = CLASSES[i]
                log_dict[f"Valid_CLASS/{class_name}"]=score.item()
            except:
                log_dict[f"Valid_CLASS/class_{i}"]=score.item()
        
        wandb.log(log_dict)

        if val_dice > best_dice:
            print(f"🎉 New Best Model! (Dice: {best_dice:.4f} -> {val_dice:.4f}) Saving...")
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            wandb.log({"Best Dice": best_dice})

    wandb.finish()

if __name__ == "__main__":
    main(args)