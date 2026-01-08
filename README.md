# Hand Bone Image Segmentation

## 1\. 프로젝트 개요

### 1.1 프로젝트 주제 및 전체 구조도

주제  
Hand Bone Image Segmentation  
X-ray 손 뼈 영상에서 총 29개 뼈 클래스를 픽셀 단위로 분할(Semantic Segmentation) 하는 모델 개발

<img width="865" height="567" alt="Image" src="https://github.com/user-attachments/assets/b60a5fa6-2480-4ff8-b6ab-d07005c420dc" />

### 1.2 프로젝트 목표 및 기법법

목표  
- 고해상도(2048 × 2048) 손 X-ray 이미지를 입력으로 사용  
- 각 픽셀을 손 뼈의 세부 구조(총 29개 클래스)에 대응  
- Dice Coefficient 기준 성능 극대화  

주요 기술  
- Semantic Segmentation  
- 의료 영상 처리  
- 앙상블 및 Pseudo Labeling  


### 1.3 프로젝트 구조도

```
pro-cv-segmentation-cv-12/
├── EDA
│   ├── eda.ipynb
│   └── prediction_analysis.ipynb
├── outputs
├── training_pipeline
│   ├── sam2unet
│   │   ├── custom_dataset.py
│   │   ├── Data_Model_Eda.ipynb
│   │   ├── heira_save.py
│   │   ├── inference.ipynb
│   │   ├── SAM2UNET.py
│   │   ├── train_for_2048.py
│   │   └── train.py
│   ├── Swin_UPerNet
│   │   ├── bone_dataset.py
│   │   ├── inference.py
│   │   ├── make_masks_and_split.py
│   │   ├── multilabel_dice_metric.py
│   │   ├── multilabel_transforms.py
│   │   ├── pack_multilabel.py
│   │   └── train.py
│   ├── EfficientNet-B3_Deeplabv3+.ipynb
│   ├── EfficientNet-B3_UNet++.ipynb
│   ├── EfficientNet-B4_UNet++.ipynb
│   ├── HRNet-W32_UNet.ipynb
│   ├── HRNet-W48+UNet++.ipynb
│   ├── sam3-unet_base.ipynb
│   ├── segformer_base_amp.ipynb
│   ├── swin_upernet-modified.ipynb
│   └── swin_upernet-pseudo.ipynb
└── utils
    ├── check_csv.ipynb
    ├── hard_voting.ipynb
    ├── make_distance_map.ipynb
    ├── pseudo_labeling_train.ipynb
    └── TTA.ipynb
```

### 1.4 개발 환경 및 협업 Tool

개발 환경  
- Hardware: GPU Server (V100 × 3)  
- OS: Linux  

협업 Tool  
- 실험 관리 및 추적: Notion, Weights & Biases (wandb)  
- 팀 커뮤니케이션: Slack, Zoom  

주요 라이브러리 및 프레임워크  
- PyTorch  
- NumPy, Pandas, tqdm  
- Albumentations  
- Segmentation Models PyTorch  
- Hugging Face  
- timm  
- ttach  


## 2. 프로젝트 팀 구성 및 역할  

<img width="320" height="320" alt="Image" src="https://github.com/user-attachments/assets/3d7aa936-8247-4686-9c78-5cd4936f794c" />

- 김범진_T8030  모델 실험, 결과 분석 및 방향 설정, Pseudo label, Class 별 Prediction 결과 분석  
- 김준수_T8048  모델 설계, 실험 효율화, 모델 고도화  
- 김한준_T8057  모델 설계 및 실험, 앙상블 전략 수립 및 실행  
- 남현지_T8061  Train Data EDA, 모델 설계 및 실험, Prediction Analysis, 앙상블 전략 설계  
- 송예림_T8107  모델 설계 및 실험, Prediction Analysis, 앙상블 전략 설계  

## 3. 프로젝트 수행 절차 및 방법

<img width="512" height="286" alt="Image" src="https://github.com/user-attachments/assets/7885c0d8-e8b9-45b6-bbe7-f4e5686cc041" />

### 3.1 프로젝트 수행 방법  

1. Dataset EDA  
  a. 데이터 정보  
  b. 데이터 구조도  
  c. EDA에서 얻은 인사이트   

2. 주제별 실험 분석    
  a. Encoder  
  b. Decoder  
  c. Input Size  
  d. Augmentation  
  e. 왼손 통합  
  f. TTA  
  g. Puseudo label  

3. 이외 접근법들   
  a. 멀티모달  
  b. 채널 입력 변경  
  c. boundary loss  
  d. 경계만 prediction 하는 head  
  e. loss 별 실험  

4. Prediction Analysis  
  a. 클래스별 성능 및 면적 분포 확인  
  b. Dice 기반 모델 간 유사도 분석  
  c. Dice 기반 클래스별 모델 유사도  

5. 앙상블 전략 구상 및 실험


### 랩업리포트 링크   


