# Fruit Classification CNN

**SKKU Introduction to Deep Learning — Assignment 2**

> 257종의 과일·채소·견과류·씨앗 이미지를 분류하는 Image Classification 과제입니다.  
> PyTorch 기반 직접 설계 CNN 모델로 Kaggle 대회에 참여합니다.

---

## Task Overview

| 항목 | 내용 |
|------|------|
| 과제 | Fruit Image Classification (257-class) |
| 평가 지표 | Categorization Accuracy |
| 플랫폼 | Kaggle (Private Leaderboard) |
| 제출 마감 | 2026년 5월 22일 (금) 23:59 KST |

---

## Dataset

Fruits-360 공개 데이터셋을 기반으로 재분할·익명화한 데이터셋입니다.

| Split | 이미지 수 | 구조 |
|-------|----------|------|
| Train | 126,040 | `train/<class_name>/*.jpg` |
| Valid | 18,023 | `valid/<class_name>/*.jpg` |
| Test | 36,016 | `test/img_NNNNNN.jpg` (라벨 없음) |

- **이미지 크기**: 100×100 RGB JPEG
- **클래스 수**: 257개 (Category 0~256)
- **클래스당 평균**: 약 700장 (균형잡힌 분포)
- **제출 형식**: `ID`(img_NNNNNN), `Category`(0~256) CSV

---

## Constraints

| 제약 | 내용 |
|------|------|
| 프레임워크 | PyTorch only |
| 학습 Epoch | **1 epoch 고정** |
| 모델 | **직접 설계한 CNN만 허용** |
| 금지 | pretrained weights 일절 금지 (ResNet, VGG, EfficientNet 등) |

---

## Project Structure

```
fruit_classification_cnn-SKKU_DL-project/
├── .cursor/
│   └── rules/
│       ├── assignment2-constraints.mdc   # 과제 제약 규칙
│       └── git-commit-convention.mdc     # 커밋 메시지 컨벤션
├── data/                                 # 데이터셋 (git 제외)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── classes.txt
├── SimpleBaseline_CNN_kaggle-2.py        # 제공된 베이스라인 코드
├── .gitignore
└── README.md
```

---

## Approach

1 epoch 제약 조건 하에서 성능을 극대화하기 위해 아래 전략을 적용합니다.

| 단계 | 기법 |
|------|------|
| 전처리 | Normalization, Resize |
| 데이터 증강 | RandomFlip, RandomRotation, ColorJitter |
| 모델 구조 | 직접 설계한 Deep CNN + BatchNorm + Dropout |
| 최적화 | AdamW + OneCycleLR Scheduler |
| 정규화 | Dropout, Weight Decay |

---

## Requirements

```bash
pip install torch torchvision tqdm pandas numpy pillow
```

---

## Usage

```bash
# 베이스라인 실행
python SimpleBaseline_CNN_kaggle-2.py

# 데이터는 ./data/ 디렉토리에 위치해야 합니다
# Kaggle 환경에서는 /kaggle/input/ 자동 감지
```

---

## Submission

- **Kaggle**: `submission.csv` 업로드 (하루 최대 10회, Private LB 반영 2회)
- **iCampus**: `학번_이름.zip` (코드 `.py`/`.ipynb` + 보고서 `.pdf`)
