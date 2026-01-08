# Imports
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import ttach as tta


# ## Constants & 설정
# 

# In[2]:


# 클래스 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

# 경로 설정
SAVED_DIR = "checkpoints"
MODEL_NAME = "efficientnetb3_unetplusplus_9736.pt"
IMAGE_ROOT = "../data/test/DCM"  # 테스트 데이터 경로를 입력하세요

# 추론 설정
BATCH_SIZE = 2
NUM_WORKERS = 2
THRESHOLD = 0.5
OUTPUT_CSV = "output_left.csv"
USE_TTA = True  # TTA 사용 여부


# ## Helper Functions
# 

# In[3]:


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    """
    RLE로 인코딩된 결과를 mask map으로 복원합니다.
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


# ## Dataset Class
# 

# In[4]:


class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, transforms=None):
        """
        추론용 Dataset 클래스
        """
        self.image_root = image_root
        
        # 이미지 파일 목록 생성
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        _filenames = np.array(sorted(pngs))
        self.filenames = _filenames
        self.transforms = transforms
        
        # 핸드 타입(Left/Right) 자동 매핑 로직
        # Test 데이터도 Train과 동일한 규칙(폴더 내 정렬 시 1번=Right, 2번=Left)을 따른다고 가정
        self.hand_side_map = {}
        
        files_by_folder = defaultdict(list)
        for fname in _filenames:
            folder = os.path.dirname(fname)
            files_by_folder[folder].append(fname)
            
        for folder, files in files_by_folder.items():
            files.sort()
            if len(files) > 0:
                self.hand_side_map[files[0]] = 'Right'
            if len(files) > 1:
                self.hand_side_map[files[1]] = 'Left'
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        
        # 원본 이미지 크기 저장 (나중에 복원 시 필요할 수 있음)
        original_shape = image.shape[:2]
        
        image = image / 255.
        
        # --- 오른손(Right)일 경우 Flip 적용 ---
        hand_side = self.hand_side_map.get(image_name, 'Unknown')
        
        if hand_side == 'Right':
            # 학습 때와 똑같이 좌우 반전시켜 모델에 넣습니다.
            image = cv2.flip(image, 1) 
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tensor
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
            
        # hand_side 정보도 같이 반환해야 나중에 결과를 다시 뒤집을 수 있습니다.
        return image, image_name, hand_side


# In[ ]:


import torch
import torch.nn.functional as F
import ttach as tta

def test_single_image(model, image_tensor, hand_side, use_tta=False):
    model.eval()
    
    img = image_tensor.unsqueeze(0).cuda()  # (1, C, H, W)
    
    # TTA 적용
    if use_tta:
        # TTA 변환 정의 (Scale 추가)
        tta_transforms = tta.Compose([
            tta.Scale(scales=[0.75, 1.0, 1.25], 
                      interpolation="bilinear"), # Multi-scale 적용
        ])
        
        # 모델을 TTA 래퍼로 감싸기
        # merge_mode='mean': 결과를 평균내어 합침
        tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')
        
        with torch.no_grad():
            output = tta_model(img)
    else:
        with torch.no_grad():
            output = model(img)
    
    # 크기 복원 (2048x2048)
    # TTA를 썼더라도 최종 output은 입력 img 크기와 같으므로, 
    # 원본 해상도(2048)로 복원하는 과정은 동일하게 유지합니다.
    #output = F.interpolate(output, size=(2048, 2048), mode="bilinear", align_corners=False)
    output = torch.sigmoid(output)
    
    # 오른손(Right)인 경우, 예측된 마스크를 다시 원래대로 뒤집음 (Unflip)
    if hand_side == 'Right':
        output = torch.flip(output, [3])  # 좌우 반전 (W 차원)
    
    return output.squeeze(0)  # (Class, H, W)
'''
'''
import torch
import torch.nn.functional as F

def test_single_image(model, image_tensor, hand_side, use_tta=False):
    model.eval()
    
    # (C, H, W) -> (1, C, H, W)
    img = image_tensor.unsqueeze(0).cuda()
    
    # 기준 사이즈 (현재 들어온 이미지의 크기)
    base_h, base_w = img.shape[2], img.shape[3]
    
    if use_tta:
        # 1. 사용할 스케일 정의 (질문하신 수치 반영)
        scales = [0.796875, 1.0, 1.203125] 
        
        # 결과를 누적할 변수 초기화
        logit_sum = 0.0
        
        with torch.no_grad():
            for scale in scales:
                # 2. 입력 이미지 리사이즈 (Scale 적용)
                # 스케일에 맞춰 타겟 H, W 계산
                target_h = int(base_h * scale)
                target_w = int(base_w * scale)
                
                # 이미지를 줄이거나 키워서 모델에 넣음
                scaled_img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # 3. 모델 추론
                output = model(scaled_img)
                
                # 4. 결과 복원 (Restore)
                # 합치기(Merge) 위해 다시 원래 base 사이즈로 되돌림
                # Logit(실수) 상태에서 Bilinear로 복원해야 경계가 부드러움
                output = F.interpolate(output, size=(base_h, base_w), mode='bilinear', align_corners=False)
                
                # 5. 누적
                logit_sum += output
        
        # 6. 평균 (Mean Merge)
        output = logit_sum / len(scales)
        
    else:
        # TTA 안 쓸 때는 그냥 추론
        with torch.no_grad():
            output = model(img)

    # -----------------------------------------------------------
    # 최종 후처리 (Resize -> Sigmoid -> Unflip)
    # -----------------------------------------------------------

    # 1. 최종 제출 크기(2048)로 복원
    # TTA 과정에서 base_size로 합쳤으므로, 마지막에 2048로 키워줍니다.
    output = F.interpolate(output, size=(2048, 2048), mode='bilinear', align_corners=False)
    
    # 2. Sigmoid 적용 (Logit -> Probability)
    output = torch.sigmoid(output)
    
    # 3. 오른손(Right)인 경우 다시 뒤집기 (Unflip)
    if hand_side == 'Right':
        output = torch.flip(output, [3])  # 좌우 반전 (W 차원)
    
    return output.squeeze(0)  # (Class, H, W)


# In[6]:


def test(model, data_loader, thr=0.5, use_tta=False):
    """
    모델을 사용하여 추론을 수행합니다 (TTA 지원).
    """
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    
    print(f"TTA 사용: {use_tta}")
    
    for step, (images, image_names, hand_sides) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # 배치를 개별 이미지로 처리 (TTA를 위해)
        for img, image_name, hand_side in zip(images, image_names, hand_sides):
            # 단일 이미지 추론 (TTA 적용)
            output = test_single_image(model, img, hand_side, use_tta=use_tta)
            
            # Threshold 적용 -> Numpy 변환 (Class, H, W)
            output = (output > thr).detach().cpu().numpy().astype(np.uint8)
            
            # 각 클래스별로 RLE 인코딩
            for c, segm in enumerate(output):
                # segm shape: (H, W)
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


# ## 체크포인트 로딩
# 

# In[7]:


# 체크포인트 경로 확인
checkpoint_path = os.path.join(SAVED_DIR, MODEL_NAME)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

print(f"체크포인트 로딩: {checkpoint_path}")
model = torch.load(checkpoint_path)
model.eval()
print("체크포인트 로딩 완료")


# ## 데이터 준비
# 

# In[8]:


# 테스트 데이터 경로 확인
if not os.path.exists(IMAGE_ROOT):
    raise FileNotFoundError(f"테스트 데이터 경로를 찾을 수 없습니다: {IMAGE_ROOT}")

print(f"테스트 데이터 경로: {IMAGE_ROOT}")

# 데이터 변환 정의
transform = A.Resize(2048, 2048)

# Dataset 및 DataLoader 생성
test_dataset = XRayInferenceDataset(image_root=IMAGE_ROOT, transforms=transform)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False
)

print(f"테스트 데이터 개수: {len(test_dataset)}")


# ## 추론 수행
# 

# In[9]:


print("추론 중...")
rles, filename_and_class = test(model, test_loader, thr=THRESHOLD, use_tta=USE_TTA)

print(f"추론 완료. 총 {len(rles)}개의 마스크 생성됨")


# ## 결과 저장
# 

# In[10]:


# CSV 파일로 저장
classes, filename = zip(*[x.split("_", 1) for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.to_csv(OUTPUT_CSV, index=False)
print(f"결과 저장 완료: {OUTPUT_CSV}")
print("=" * 50)
print("추론 완료!")
print("=" * 50)


# In[ ]:




