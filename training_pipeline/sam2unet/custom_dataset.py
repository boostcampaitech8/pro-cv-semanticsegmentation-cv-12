import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# 클래스 정의는 전역으로 둬도 괜찮습니다.
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None):
        """
        image_root: 이미지가 있는 폴더 경로 (train.py에서 받아옴)
        label_root: 라벨(json)이 있는 폴더 경로
        """
        self.image_root = image_root
        self.label_root = label_root
        # 1. 파일 리스트 읽기 (함수 안에서 실행되므로 안전함)
        # png 파일만 골라내기
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        jsons = {
            os.path.relpath(os.path.join(root, fname), start=label_root)
            for root, _dirs, files in os.walk(label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # --- [추가 1] 핸드 타입(Left/Right) 자동 매핑 로직 ---
        # 규칙: 같은 ID 폴더 내에서 파일명 정렬 시 1번째=Right, 2번째=Left
        self.hand_side_map = {}
        
        # 1. 폴더별로 파일 그룹화
        files_by_folder = defaultdict(list)
        for fname in _filenames:
            folder = os.path.dirname(fname)
            files_by_folder[folder].append(fname)
            
        # 2. 정렬 후 Right/Left 할당
        for folder, files in files_by_folder.items():
            files.sort() # 이름순 정렬
            if len(files) > 0:
                self.hand_side_map[files[0]] = 'Right' # 첫 번째는 오른손
            if len(files) > 1:
                self.hand_side_map[files[1]] = 'Left'  # 두 번째는 왼손
        # 3장 이상인 경우 등 예외처리는 필요 시 추가
        
        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]
        
        gkf = GroupKFold(n_splits=20)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

        self.clahe1 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(64, 64))
        self.clahe2 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(64, 64))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        # 1. 이미지 읽기
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        ch1 = image_gray # 그레이 스케일 이미지
        #ch2 = self.clahe1.apply(image_gray)
        #ch3 = self.clahe2.apply(image_gray)

        #ch2 = cv2.Canny(ch1, 100, 200)
        #sobel_x = cv2.Sobel(ch1, cv2.CV_64F, 1, 0, ksize=3, scale=2)
        #sobel_y = cv2.Sobel(ch1, cv2.CV_64F, 0, 1, ksize=3, scale=2)
        #ch2 = cv2.magnitude(sobel_x, sobel_y)
        #mask1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        #mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        #ch2 = cv2.filter2D(ch1, -1, mask1)
        #ch3 = cv2.filter2D(ch1, -1, mask2)        
        #ch3 = cv2.Laplacian(ch1, cv2.CV_64F)

        #experimet5
        """
        img_blur = cv2.GaussianBlur(ch1, (3, 3), 0)
        grad_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)

        ch2 = cv2.magnitude(grad_x, grad_y)
        ch3 = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)

        min_val, max_val = ch2.min(), ch2.max()
        if max_val - min_val > 1e-5:
            ch2 = (ch2 - min_val) / (max_val - min_val) * 255.0
        else:
            ch2 = np.zeros_like(ch2)

        # ch3 정규화 (Laplacian은 음수도 나오므로 절대값 후 정규화 추천하거나, 그냥 정규화)
        # 여기서는 분포 유지를 위해 Min-Max만 적용
        min_val, max_val = ch3.min(), ch3.max()
        if max_val - min_val > 1e-5:
            ch3 = (ch3 - min_val) / (max_val - min_val) * 255.0
        else:
            ch3 = np.zeros_like(ch3)
        """
       
        image = np.stack([ch1, ch1, ch1], axis=-1)
        image = image.astype(np.float32) / 255.0 
        
        # 2. 라벨 읽기 및 마스크 생성
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        mask = np.zeros(label_shape, dtype=np.uint8)

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # (H, W) 크기의 마스크 생성 (0=배경, 1~29=뼈)
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # class_ind + 1을 해서 0번(배경)과 구분합니다.
            cv2.fillPoly(class_label, [points], 1)
            mask[..., class_ind] = class_label
        
        # --- [추가 2] 오른손(Right)일 경우 Flip 적용 ---
        # 현재 이미지가 'Right'라면 좌우 반전시켜 'Left'처럼 만듭니다.
        hand_side = self.hand_side_map.get(image_name, 'Unknown')
        
        if hand_side == 'Right':
            # 1: Horizontal Flip (좌우 반전)
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        
        # 3. 전처리 (Transforms)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 4. 텐서 변환 (채널 순서 변경)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
            
        return {'image': image, 'label': mask, 'image_name': self.filenames[item]}

# 테스트용 코드 (직접 실행할 때만 작동)
if __name__ == "__main__":
    # 테스트 할 때 경로가 맞는지 확인 필요
    root_img = "./data/train/DCM"
    root_lbl = "./data/train/outputs_json"
    
    if os.path.exists(root_img):
        train_ds = XRayDataset(root_img, root_lbl, is_train=True)
        valid_ds = XRayDataset(root_img, root_lbl, is_train=False)
        print(f"Train size : {len(train_ds)}, Valid size : {len(valid_ds)}")
        index = 0
        print(f"Train data ID : {train_ds[index]['image_name'][2:5]}")
        print(f"Train data label shape: {train_ds[index]['label'].shape}")
        print(f"Train data image shape: {train_ds[index]['image'].shape}")
    else:
        print("경로가 없어서 테스트를 건너뜁니다.")