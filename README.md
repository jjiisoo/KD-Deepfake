# 딥페이크 탐지를 위한 지식 증류 기반 경량화 모델

**2025 한국소프트웨어공학학술대회 참가**

---

## 📰 연구 배경
딥페이크(Deepfake)는 AI를 이용해 이미지, 영상, 음성을 조작하여 실제와 유사한 콘텐츠를 만드는 기술로 최근 오픈소스 프로그램 확산으로 누구나 쉽게 제작 가능해지면서 **사회적·윤리적 문제**가 증가하고 있다.  
경찰청 통계에 따르면, 딥페이크 관련 범죄는 2021년 156건에서 2024년 7월 기준 297건으로 증가했다. 특히 성 착취물 제작 등 악용 사례가 문제이며, 유포된 콘텐츠는 완전 삭제가 어려워 **2차·3차 피해**로 이어질 수 있다.  
본 연구는 **경량화된 모델**과 **지식 증류(Knowledge Distillation)** 기법을 활용해 딥페이크 얼굴 이미지를 실시간으로 탐지하고, 피해 확산을 줄이는 것을 목표로 한다.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Librosa](https://img.shields.io/badge/Librosa-000000?style=for-the-badge&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📰 데이터 구성

### 1️⃣ 데이터 수집
- *Kaggle – Deepfake-dataset (140k+dataset real or fake)* (일부만 사용)    [데이터 링크](https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset)

### 2️⃣ 데이터 전처리
1. **파일 형식 변환:** `mp3` → `wav`  
2. **노이즈/무음 제거:** noisereduce + pydub 사용  
3. **길이 통일:** 3초 단위  
4. **데이터 분할:** Train / Validation / Test

#### 폭력 데이터
| 구분 | Train | Validation | Test |
|------|-------|------------|------|
| 개수 | 1,014 | 370        | 67   |

#### 비폭력 데이터
| 구분 | Train | Validation | Test |
|------|-------|------------|------|
| 개수 | 832   | 244        | 76   |
