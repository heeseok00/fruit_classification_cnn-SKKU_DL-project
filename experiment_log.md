# Experiment Log — Fruit Classification CNN

> **과제 제약**: PyTorch only / n_epochs = 1 / 외부 pre-trained 모델 금지 / 커스텀 CNN
>
> **평가 기준**: Kaggle Public Score (Categorization Accuracy)

---

## 실험 결과 요약

| 버전 | 주요 변경사항 | Train Acc | Valid Acc | Kaggle Public Score |
|------|--------------|-----------|-----------|---------------------|
| baseline (v0) | 원본 코드 (10 epoch, 정규화 없음) | — | — | — |
| v1 | 필수 개선: 1 epoch, augmentation, normalization, OneCycleLR | — | — | **0.84311** |
| v2 | 모델 교체: ResBlock + GAP | 0.507 | 0.944 | **0.94642** |
| v3 | batch_size 64 → 128 | 0.487 | 0.903 | 0.90627 |
| (참고) v4_batch256 | batch_size 64 → 256 (중단됨) | — | — | 0.87522 |
| v4 | **Label Smoothing(0.1)** 적용 | 0.530 | **0.950** | **0.95502** ⬆ NEW BEST |

> Kaggle Public Score를 받은 후 해당 셀을 업데이트하세요.

---

## 버전별 상세 내역

### v0 — Original Baseline
- **파일**: `SimpleBaseline_CNN_kaggle-2.py`, `baseline_code_v0.py`
- **모델 구조**:
  - 5× Conv2d → BatchNorm → ReLU → MaxPool2d
  - Flatten → Linear(8192→1024) → Linear(1024→512) → Linear(512→257)
- **학습 설정**:
  - `n_epochs = 10`, `batch_size = 64`, `lr = 0.0003` (Adam)
  - 스케줄러 없음
- **데이터 전처리**: Resize(128×128) + ToTensor 만 적용 (augmentation/normalization 없음)
- **결과**: 미제출

---

### v1 — 필수 개선 (Augmentation + OneCycleLR)
- **파일**: `baseline_code_v1.py`, `v1_augment_normalize_onecyclelr.ipynb`
- **v0 대비 변경사항**:
  - `n_epochs = 1` (과제 제약 준수)
  - Train augmentation 추가: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(20)`, `ColorJitter`
  - `transforms.Normalize(ImageNet mean/std)` 추가 (train/test 모두)
  - `OneCycleLR` 스케줄러 추가 (`max_lr=0.01`)
  - `num_workers=2` 적용
- **결과**:
  - Kaggle Public Score: **0.84311**

---

### v2 — ResBlock + Global Average Pooling
- **파일**: `v2_resblock_gap.py`, `v2_resblock_gap.ipynb`
- **v1 대비 변경사항**:
  - 모델 구조 전면 교체
    - 기존: 5 Conv + 3 FC → 신규: Stem + 4 Stage(ResBlock×2) + GAP + FC(1개)
    - `ResBlock`: 직접 구현한 Residual Block (skip connection, BatchNorm)
    - `AdaptiveAvgPool2d(1)` (Global Average Pooling)으로 파라미터 대폭 감소
  - 데이터 전처리, 옵티마이저, 스케줄러 등 나머지는 v1과 동일
- **결과**:
  - Train Acc: 0.507, Valid Acc: 0.944
  - Kaggle Public Score: **0.94642** (+0.103)
- **분석**: valid_acc(0.944) >> train_acc(0.507) → underfitting 상태. 1 epoch 특성상 모델이 데이터를 충분히 학습하지 못함

---

### (참고) v4_batch256 — batch_size 256 (중단된 실험)
- **v2 대비 변경사항**: `batch_size`: 64 → **256**
- **결과**: Kaggle Public Score: **0.87522**
- **분석**: batch 클수록 1-epoch에서 손해 명확히 확인 (v2: 0.946 > v3: 0.906 > batch256: 0.875)

---

### v3 — batch_size 128
- **파일**: `v3_batch128.py`
- **v2 대비 변경사항**:
  - `batch_size`: 64 → **128**
- **결과**:
  - Train Acc: 0.487, Valid Acc: 0.903
  - Kaggle Public Score: **0.90627**
- **분석**: batch_size가 커지면 epoch당 gradient update 횟수(step 수)가 줄어들어 1-epoch에서 오히려 불리. v2(valid 0.944)보다 valid acc 감소

---

### v4 — Label Smoothing(0.1)
- **파일**: `v4_label_smoothing.py`
- **전략 근거**: v3(batch=128)이 v2(batch=64)보다 낮음 → batch_size는 64가 최적. 257클래스의 유사한 과일 분류에서 모델 overconfidence 완화 필요
- **v2 대비 변경사항**:
  - `CrossEntropyLoss(label_smoothing=0.1)` 적용 (soft label로 일반화 개선)
  - 나머지(batch_size=64, max_lr=0.01, 모델 구조) v2와 동일
- **결과**:
  - Train Acc: 0.530, Valid Acc: **0.950** (+0.006 vs v2)
  - Kaggle Public Score: **0.95502** (+0.0086 vs v2, NEW BEST)
- **분석**: label smoothing이 효과적. valid_acc가 v2(0.944)→v4(0.950)로 향상. Kaggle에서도 0.946→0.955로 개선 확인

---

## 주요 인사이트

### 1 Epoch 학습의 특성
- 1 epoch에서는 **batch_size를 줄이는 것**이 유리 (더 많은 gradient update)
- OneCycleLR은 1 epoch에서 특히 효과적 — warmup → peak → cooldown을 1 epoch에 압축

### 모델 구조 선택
- ResBlock + GAP 조합이 기본 CNN 대비 압도적 성능 향상 (+10.3%)
- GAP은 Flatten 대비 파라미터 수를 크게 줄여 빠른 수렴에 도움

### 관찰된 패턴
- v2에서 train_acc(0.507) << valid_acc(0.944): 전형적인 **1-epoch underfitting**
  - valid set은 augmentation 없이 평가 → train 대비 "쉬운" 조건
  - train_acc가 낮다고 해서 모델이 나쁜 것이 아님

---

## 제출 파일 목록

| 파일명 | 버전 | 상태 |
|--------|------|------|
| `submission_v1_augment_normalize_onecyclelr_0.84311.csv` | v1 | 제출 완료 |
| `submission_v2_resblock_gap_0.94642.csv` | v2 | 제출 완료 |
| `submission_v3_batch128.csv` | v3 | 생성 완료 (미제출) |
| `submission_v4_label_smoothing.csv` | v4 | 생성 완료 (미제출) |
