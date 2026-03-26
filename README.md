# SketchGPT — 실제 구현 모델

## 개요

이는 **SketchGPT** 논문 [[arXiv:2405.03099](https://arxiv.org/abs/2405.03099)]을 기반으로 **PyTorch**로 구현한 스케치 생성 모델입니다.

논문에서 명시하지 않은 여러 파라미터들(예: primitive 크기, 토큰 최대 길이, pre-training/fine-tuning 에포크 수 등)은 **광범위한 실험과 파라미터 조정**을 통해 최적화되었습니다. **모델의 기본 뼈대와 핵심 수식은 논문과 일치**하지만, 일부 값과 기술적 표현은 실제 구현 시 실용성 고려로 조정되었을 수 있습니다.

---

## 논문 구현 내용

### 핵심 구조 (논문 3.3절, 3.4절)

| 단계 | 설명 | 논문 참조 |
|------|------|----------|
| **Pre-training** | 10개 전체 클래스 혼합 데이터로 단일 모델 학습 <br/> → 클래스 구분 없음 | 3.3절 Next-Token Prediction |
| **Fine-tuning** | 각 클래스별 독립적으로 fine-tune <br/> → 10개의 클래스 전용 모델 생성 | 3.4절 Per-class Models |
| **생성** | 해당 클래스의 전용 모델만 사용 <br/> → 클래스 혼합 없음 | Fig. 5 캡션 |

---

## 기술 상세

### 1. 데이터 처리

- **원본 데이터**: [Google QuickDraw](https://quickdraw.withgoogle.com/data) ndjson 형식
  - 자동 다운로드 및 캐시 지원
  - 각 클래스당 최대 7,000개 샘플 (학습/검증/테스트 분할)

- **전처리**:
  - Stroke 3-tuple 변환: `[dx, dy, pen_lift]`
  - 정규화: x, y 좌표를 [0, 1] 범위로 독립 정규화

### 2. 기본 구성 요소

#### Primitive 기반 토크나이저
- **Primitive 개수**: EDA를 통해 결정 (기본 16개)
  - 각 primitive는 균등하게 분포한 방향 벡터 (0도 ~ 360도)
  - [cos(θ), sin(θ)] 형태

- **토큰화**:
  1. 각 점(dx, dy)을 가장 가까운 primitive 방향으로 매핑
  2. 거리 L = ||(dx, dy)||에 따라 토큰 반복 (1~8회)
  3. Pen lift 지점에서 `TOKEN_SEP` 삽입
  4. 시작/종료: `TOKEN_BOS`, `TOKEN_EOS`
  5. Padding: `TOKEN_PAD`

```python
# 예: [dx=0.01, dy=0.01, lift=0] 
#    → primitive_id=0 (45도 방향)
#    → 거율 계산: L ≈ 0.014
#    → 반복 횟수: ceil(0.014 / 0.01) = 2회
#    → 토큰: [PRIM_0, PRIM_0]
```

#### Transformer 언어 모델
```
SketchGPT
├─ Token Embedding (vocab_size=20, d_model=512)
├─ Positional Embedding (max_seq=256)
├─ 8× TransformerBlock
│  ├─ Causal Self-Attention (8 heads, d_head=64)
│  ├─ MLP (d_ff=2048, GELU)
│  └─ LayerNorm + Residual
└─ Language Model Head (vocab_size 예측)

총 파라미터: ~35.8M
```

- **주요 특성**:
  - Causal masking: 이전 토큰만 참조
  - Mixed Precision (AMP) 활성화로 메모리 효율
  - Gradient clipping: 1.0

### 3. 학습 전략

#### Pre-training (모든 클래스 혼합)

| 파라미터 | 값 | 설정 목적 |
|---------|-----|----------|
| 에포크 | 15 | 전체 데이터 충분 학습 |
| 배치 크기 | 64 | GPU 메모리 (RTX 3090) 최적화 |
| 학습률 | 0.001 | Adam 기본 교과서 값 |
| 스케줄러 | Cosine Annealing | 학습율 점진 감소 |
| Early Stopping | 3 에포크 | 검증 손실 정체 시 조기 종료 |

- **손실 함수**: Cross-entropy NLL (language modeling)
  - Padding 토큰 무시

#### Fine-tuning (클래스별 독립)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 에포크 | 10 | pre-train보다 짧음 |
| 학습률 | 0.0001 | pre-train의 1/10 |
| 데이터 | 클래스 단일 | 각 클래스 전용 학습 |
| Partial Sketches | 10%~90% 범위 | Completion task 시뮬레이션 |

**Partial Sketch Dataset**: 
- 학습 시 각 스케치의 10%~90% 구간을 랜덤하게 선택
- 모델이 부분 스케치 완성(completion) 능력 학습
- 생성 시 실제 스케치의 50% 프롬프트 제공

### 4. 생성 (Decoding)

```python
generate(
    prompt=[실제 스케치 처음 50%],  # PROMPT_RATIO=0.5
    max_new_tokens=200,
    temperature=1.0,                # 다양성 제어
    top_k=10,                       # Top-K 샘플링
    min_new_tokens=20,              # 최소 생성 길이
)
```

- **샘플링 전략**: Top-K + Temperature
  - 다양한 생성 결과 확보
  - `TOKEN_EOS` 도달 시 자동 종료

---

## 핵심 클래스 구조

### 데이터 클래스

#### 1. **SketchDataset**
```python
class SketchDataset(Dataset):
    """Pre-training & 평가용 전체 시퀀스 데이터셋."""
    def __init__(self, tokens_list, labels, max_seq):
        # 각 스케치를 max_seq 길이로 패딩/트림
        # tokens: [TOKEN_BOS, ..., TOKEN_EOS, TOKEN_PAD, ...]
        
    def __getitem__(self, idx):
        return (tokens: Tensor[max_seq], label: int)
```

#### 2. **PartialSketchDataset**
```python
class PartialSketchDataset(Dataset):
    """Fine-tuning용 부분 스케치 데이터셋 (Completion task)."""
    def __init__(self, tokens_list, labels, max_seq, 
                 min_ratio=0.1, max_ratio=0.9):
        # 각 스케치의 10%~90% 구간을 랜덤하게 선택
        # 모델이 부분 입력에서 완성 학습
        
    def __getitem__(self, idx):
        # 원본 길이의 10~90% 지점에서 자름
        partial_tokens = toks[:cut_point]  # cut ∈ [min_ratio*n, max_ratio*n]
        return (partial_tokens_padded, label)
```

### 모델 클래스

#### 3. **CausalSelfAttention**
```python
class CausalSelfAttention(nn.Module):
    """인과 마스킹을 갖춘 다중 헤드 자기 어텐션."""
    def __init__(self, d_model, n_heads, max_seq, dropout):
        self.qkv = Linear(d_model, 3*d_model)        # Q, K, V 생성
        self.proj = Linear(d_model, d_model)         # 투영
        self.causal_mask = tril(ones(max_seq, max_seq))  # 마스킹
        
    def forward(self, x: Tensor[B, L, D]) -> Tensor[B, L, D]:
        q, k, v = split_qkv(x)  # 각 [B, n_heads, L, d_head]
        scores = (q @ k.T) / sqrt(d_head)  # 어텐션 점수
        scores = scores.masked_fill(causal_mask==0, -inf)  # 인과성
        attn_weights = softmax(scores, dim=-1)
        return proj(attn_weights @ v)
```

#### 4. **TransformerBlock**
```python
class TransformerBlock(nn.Module):
    """표준 트랜스포머 블록 (Self-Attn + MLP + Residual)."""
    def __init__(self, d_model, n_heads, d_ff, max_seq, dropout):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(...)
        self.ln2 = LayerNorm(d_model)
        self.mlp = Sequential(
            Linear(d_model, d_ff),
            GELU(),
            Linear(d_ff, d_model),
            Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Residual + Pre-norm
        x = x + self.mlp(self.ln2(x))
        return x
```

#### 5. **SketchGPT**
```python
class SketchGPT(nn.Module):
    """메인 언어 모델 - 스케치 토큰 시퀀스 생성."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, 
                 d_ff, max_seq, dropout):
        self.tok_emb = Embedding(vocab_size, d_model)      # 토큰 임베딩
        self.pos_emb = Embedding(max_seq, d_model)         # 위치 임베딩
        self.blocks = ModuleList([
            TransformerBlock(...) for _ in range(n_layers)  # 8개 블록
        ])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)         # 출력 헤드
        
    def forward(self, tokens: Tensor[B, L]) -> Tensor[B, L, vocab_size]:
        # tokens: [BOS, tok1, tok2, ..., EOS, PAD, ...]
        # 출력: 각 위치에서 다음 토큰 확률 분포
```

---

## 주요 수식

### 1. **Attention 메커니즘**

#### Scaled Dot-Product Attention
$$\operatorname{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q, K, V$: Query, Key, Value 행렬 (길이 $L$, 차원 $d_k = d_{model}/n_{heads}$)
- $d_k = 512/8 = 64$
- 인과 마스킹: future tokens를 $-\infty$로 설정

#### Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\operatorname{head}\_1, ..., \operatorname{head}\_8)W^O$$
$$\operatorname{head}\_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- 8개 병렬 어텐션 헤드
- 각 헤드는 독립적인 가중치 행렬 사용

### 2. **Language Modeling 손실**

#### Next-Token Prediction Loss (NLL)
$$\mathcal{L}_{NTP} = -\sum_{t=1}^{L-1} \log P(x_t | x_{<t}; \theta)$$

여기서 $P(x_t | x_{<t})$는 모델의 예측 확률:
$$P(x_t | x_{<t}) = \text{softmax}(\operatorname{SketchGPT}(x_{<t}))[x_t]$$

**구현**:
```python
logits = model(tokens)  # [B, L, vocab_size]
loss = F.cross_entropy(
    logits[:, :-1].view(-1, vocab_size),  # 예측 (t=0...L-2)
    tokens[:, 1:].view(-1),                # 실제값 (t=1...L-1)
    ignore_index=TOKEN_PAD
)
```

- Padding 토큰 무시 (`ignore_index=3`)
- Per-token 평균 손실

### 3. **Weighted Sampling (Class Imbalance)**

클래스별 데이터 불균형 해결:
$$w_c = \frac{1}{n_c}$$

여기서 $n_c$는 클래스 $c$의 학습 샘플 수.

Pre-training에서 `WeightedRandomSampler` 사용:
```python
# 클래스별 비중 동등화
class_weights = [1.0/count for count in per_class_counts]
sampler = WeightedRandomSampler(
    weights=[class_weights[label] for label in labels],
    num_samples=len(dataset)
)
```

### 4. **Primitive 기반 Tokenization**

#### Primitive 빌드
$$P = \{\mathbf{p}_k : \mathbf{p}_k = [\cos(2\pi k/16), \sin(2\pi k/16)]\}_{k=0}^{15}$$

- $N_{primitives} = 16$개 균등 분포 벡터
- 각 벡터는 단위 벡터 (norm = 1)

#### Primitive ID 계산
$$\operatorname{prim\_id}(dx, dy) = \arg\max_k \left(\frac{\mathbf{p}_k \cdot [dx, dy]}{||[dx, dy]||}\right)$$

- $[dx, dy]$를 정규화한 후 가장 가까운 primitive 방향 선택

#### Scale Factor (반복 횟수)
$$s(dx, dy) = \max(1, \min(8, \lceil L / \lambda \rceil))$$
$$L = \sqrt{dx^2 + dy^2}, \quad \lambda = 0.01 \text{ (PRIM\_LENGTH)}$$

예시:
- $L = 0.005$: $s = 1$ (1회 반복)
- $L = 0.01$: $s = 1$ (1회)
- $L = 0.02$: $s = 2$ (2회)
- $L = 0.08$: $s = 8$ (8회, 최대값)

### 5. **정규화 (Normalization)**

#### 절대 좌표 → 상대 좌표 변환
각 stroke 시퀀스 $s3 = [[x_0, y_0], [x_1, y_1], ..., [x_n, y_n]]$를 다음과 같이 변환:

$$x_{\operatorname{abs}}^{(i)} = x_0 + \sum_{j=0}^{i-1} dx_j, \quad y_{\operatorname{abs}}^{(i)} = y_0 + \sum_{j=0}^{i-1} dy_j$$

#### 좌표 정규화 (각 사각형 [0, 1]×[0, 1])
$$x'_i = \frac{x_{\operatorname{abs}}^{(i)} - \min_j x_{\operatorname{abs}}^{(j)}}{\max_j x_{\operatorname{abs}}^{(j)} - \min_j x_{\operatorname{abs}}^{(j)} + \epsilon}$$

비슷하게 $y'_i$도 정규화.

#### 상대 좌표 재계산
$$\Delta x'_i = x'_i - x'_{i-1}, \quad \Delta y'_i = y'_i - y'_{i-1}$$

**특징**: x, y 좌표를 독립적으로 정규화하여 aspect ratio 무관

### 6. **학습률 스케줄**

#### Cosine Annealing
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

- $\eta_0 = 0.001$ (pre-train), $0.0001$ (fine-tune)
- $\eta_{\min} = 0.1 \times \eta_0$
- $T = 15$ (pre-train epochs), $10$ (fine-tune epochs)

### 7. **Gradient Clipping**
$$\|\nabla_\theta \mathcal{L}\| > \operatorname{clip\_norm}(=1.0) \Rightarrow \nabla_\theta \mathcal{L} \leftarrow \operatorname{clip\_norm} \cdot \frac{\nabla_\theta \mathcal{L}}{\|\nabla_\theta \mathcal{L}\|}$$

Exploding gradient 방지

---

## 파라미터 최적화 (실험 기반)

### EDA (Exploratory Data Analysis)에서 결정된 값

```python
# 원본 모든 스케치의 stroke 분석
N_PRIMITIVES = 16          # 방향 벡터 개수
PRIM_LENGTH = 0.01         # 중앙값 기반 스케일
MAX_SEQ = 256              # 95 percentile 토큰 길이 (2의 거듭제곱)
```

**EDA 시각화**:
1. **Stroke Direction Distribution**: Primitive 방향 균등성 확인
2. **Stroke Length Distribution**: PRIM_LENGTH 결정
3. **Token Length Distribution**: MAX_SEQ, 배치 크기 결정

### 예시 값 (10개 클래스 평균)
```
N_PRIMITIVES: 16
PRIM_LENGTH: 0.01 (median stroke length)
MAX_SEQ: 256 (95th percentile 토큰 길이)
```

---

## 체크포인트 구조

```
checkpoints/
├── eda_params.json                    # EDA 결과 (N_PRIMITIVES, MAX_SEQ, etc)
├── pt_best.pt                         # 공통 Pre-train (1개)
├── gen_airplane.pt                    # Airplane class fine-tune
├── gen_bus.pt
├── gen_canoe.pt
├── gen_car.pt
├── gen_helicopter.pt
├── gen_hot_air_balloon.pt
├── gen_motorbike.pt
├── gen_sailboat.pt
├── gen_submarine.pt
└── gen_train.pt                       # (총 10개 클래스)
```

---

## 설치 및 실행

### 요구사항

- **Python**: 3.8+
- **GPU**: RTX 3090 권장 (최소 8GB VRAM)
- **CUDA**: 12.1 호환

### 설치

```bash
# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 추가 라이브러리
pip install numpy matplotlib tqdm requests scikit-learn Pillow scipy
```

### 실행 옵션

```python
# 처음부터 전부 실행 (EDA ~ 생성)
python sketchgpt_final.py
# (내부적으로 main() 호출)

# 또는 각 단계를 제어하여 실행
main(skip_eda=False, skip_pretrain=False, skip_finetune=False)

# Pre-train 재사용, Fine-tune만 새로 실행
main(skip_eda=True, skip_pretrain=True, skip_finetune=False)

# 모든 가중치 재사용, 시각화만 실행 (기존 checkpoints 필수)
main(skip_eda=True, skip_pretrain=True, skip_finetune=True)
```

**예상 실행 시간**:
- EDA: ~10분 (500 샘플/클래스)
- Pre-training: ~30분 (15 에포크, 배치 64)
- Fine-tuning: ~20분 (10 에포크 × 10 클래스)
- 생성 및 시각화: ~10분

---

## 출력 결과

### 디렉토리 구조

```
outputs/<timestamp>/
├── eda_analysis.png                   # Primitive/길이 분포
├── raw_samples.png                    # QuickDraw 원본 4개/클래스
├── generated_sketches.png             # Original vs Generated 비교 (큰 그리드)
├── pretrain_loss.png                  # Pre-train 손실 곡선
├── ft_loss_*.png                      # Fine-tune 손실 곡선 (클래스별)
└── sequential/                        # Stroke 누적 이미지
    ├── overview.png                   # 100개 전체 요약
    ├── airplane/
    │   ├── preview.png                # 해당 클래스 10개 샘플 미리보기
    │   ├── sample_00/
    │   │   ├── stroke_001.png         # 첫 번째 stroke
    │   │   ├── stroke_002.png         # 누적 (첫+두 번째)
    │   │   ├── ...
    │   │   ├── stroke_all.png         # 완성본
    │   │   ├── strokes.json           # 좌표 데이터
    │   │   └── strokes.npy            # 수치 배열 형식
    │   ├── sample_01/
    │   └── ...
    ├── bus/
    ├── canoe/
    └── ... (10개 클래스)
```

### 주요 시각화

1. **EDA Analysis** (`eda_analysis.png`)
   - Stroke 방향 분포 (히스토그램)
   - Stroke 길이 분포
   - 토큰 길이 분포 (95 percentile 표시)

2. **Generated Sketches** (`generated_sketches.png`)
   - 위행: 원본 스케치 (QuickDraw)
   - 아래행: 생성된 스케치 (공식 모델)
   - 그리드: 클래스 × 4 샘플

3. **Sequential Overview** (`sequential/overview.png`)
   - 10개 클래스 × 10 샘플 = 100개 생성 완성본
   - 한 장의 이미지에 전체 결과 요약

4. **Per-class Preview** (`sequential/<class>/preview.png`)
   - 해당 클래스의 10개 샘플 미리보기

5. **Stroke Progression** (`sequential/<class>/sample_XX/stroke_*.png`)
   - 스케치 그리기 과정의 단계별 이미지
   - 예: stroke_001.png → stroke_002.png → ... → stroke_all.png

---

## 논문과의 차이점

### 동일한 부분 ✓
- **전체 구조**: Pre-training (혼합) → Fine-tuning (클래스별) → 생성
- **모델 아키텍처**: Transformer with causal attention
- **기본 수식**: Next-token prediction, Language modeling loss
- **Primitive 기반 토크나이저** (논문 3.2절)
- **학습 전략**: Weighted sampling (클래스 불균형 해결)

### 실험 기반 조정된 부분 ⚠️
| 항목 | 논문 | 구현 | 사유 |
|------|------|------|------|
| Pre-train 에포크 | 미공개 | 15 | 검증 손실 수렴 기준 |
| Fine-tune 에포크 | 미공개 | 10 | Early stopping 효율성 |
| 배치 크기 | 미공개 | 64 | RTX 3090 VRAM (24GB) 최적화 |
| 학습률 | 미공개 | 0.001 (PT), 0.0001 (FT) | Adam 표준 설정 + 크기 조정 |
| 토큰 최대 길이 | 미공개 | 256 | EDA 기반 (95 percentile) |
| Primitive 개수 | 미공개 | 16 | EDA 방향 분포 분석 |

### 선택적 변형 (논문 정신 유지)
- **Causal Masking**: 표준 트랜스포머 (논문에서 명시 안 함)
- **Positional Encoding**: Absolute positional embeddings
- **Partial Sketches**: Pre-training과 구분되는 fine-tuning 기법

---

## 추가 정보

### 클래스 목록 (10개)

```python
CLASSES = [
    "airplane",           # 비행기
    "bus",               # 버스
    "canoe",             # 카누
    "car",               # 자동차
    "helicopter",        # 헬리콥터
    "hot air balloon",   # 풍선
    "motorbike",         # 오토바이
    "sailboat",          # 돛단배
    "submarine",         # 잠수함
    "train"              # 기차
]
```

**데이터 분할** (클래스당):  
```
Pre-train 용: 5,000개
  ├─ Train (학습):   ~2,500개 (50%)
  ├─ Val (검증):     ~1,250개 (25%)
  └─ Test (테스트):  ~1,250개 (25%)

Fine-tune 용: 각 클래스별 독립
  ├─ Train (학습):   ~3,350개 (67%)
  └─ Val (검증):     ~1,650개 (33%)
```

### 주요 상수 값

| 상수명 | 값 | 의미 |
|-------|-----|------|
| `TOKEN_BOS` | 0 | 시작 토큰 (Begin of Sequence) |
| `TOKEN_EOS` | 1 | 종료 토큰 (End of Sequence) |
| `TOKEN_SEP` | 2 | Separator (펜 리프트, stroke 경계) |
| `TOKEN_PAD` | 3 | 패딩 토큰 (패딩 위치) |
| `SPECIAL_TOKENS` | 4 | 특수 토큰 개수 |
| `VOCAB_SIZE` | 20 | 전체 어휘 크기 (4 + 16 primitives) |
| `N_PRIMITIVES` | 16 | Primitive 개수 (방향 벡터) |
| `PRIM_LENGTH` | 0.01 | Primitive 스케일 (정규화 stroke 길이) |
| `MAX_SEQ` | 256 | 최대 시퀀스 길이 (토큰) |
| `MAX_SEQ_HARD_LIMIT` | 512 | 절대 최대값 (메모리 보호) |

### 하이퍼파라미터 요약

| 카테고리 | 파라미터 | 값 |
|---------|---------|-----|
| **모델** | Layers | 8 |
| | Attention Heads | 8 |
| | Hidden Dim | 512 |
| | Feed-forward Dim | 2048 |
| | Dropout | 0.1 |
| | Max Seq Length | 256 |
| **Pre-train** | Batch Size | 64 |
| | Epochs | 15 |
| | Learning Rate | 1e-3 |
| | Optimizer | AdamW (β1=0.9, β2=0.95) |
| | Weight Decay | 0.01 |
| **Fine-tune** | Batch Size | 64 |
| | Epochs | 10 |
| | Learning Rate | 1e-4 |
| | Partial Ratio | 10%~90% |
| **Generation** | Temperature | 1.0 |
| | Top-K | 10 |
| | Min New Tokens | 20 |
| | Prompt Ratio | 50% |

### 기술 스택
- **Framework**: PyTorch 2.x
- **Optimization**: Automatic Mixed Precision (AMP)
- **GPU Optimization**: cuBLAS TF32, cudnn.benchmark
- **Data**: Google QuickDraw ndjson format
- **Visualization**: Matplotlib, PIL, Scipy (spline interpolation)

### 주요 함수 목록

| 함수 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `download_ndjson(class_name, max_n)` | QuickDraw 데이터 다운로드 | 클래스명, 최대 샘플 수 | 스케치 리스트 |
| `drawing_to_stroke3(drawing)` | 드로잉 → Stroke 3-tuple | [[[x], [y]], ...] | np.ndarray [N, 3] |
| `normalize_stroke3(s3)` | 좌표 정규화 [0,1] | np.ndarray [N, 3] | 정규화된 배열 |
| `build_primitives(n)` | Primitive 벡터 생성 | n=16 | np.ndarray [16, 2] |
| `prim_id(dx, dy)` | 가장 가까운 primitive 찾기 | dx, dy | 0~15 (primitive ID) |
| `scale_factor(dx, dy)` | 토큰 반복 횟수 계산 | dx, dy | 1~8 |
| `tokenize(s3)` | Stroke → 토큰 시퀀스 | np.ndarray [N, 3] | list of tokens |
| `run_eda(classes, n_sample)` | EDA 분석 및 가시화 | 클래스 리스트 | (N_PRIMITIVES, MAX_SEQ, PRIM_LENGTH) |
| `make_loader(dataset, batch_size)` | DataLoader 생성 | Dataset, int | DataLoader |
| `make_model()` | SketchGPT 모델 초기화 | - | SketchGPT 인스턴스 |
| `pretrain(model, train_ds, val_ds)` | Pre-training 실행 | 모델, 데이터셋 | pre-trained 모델 |
| `finetune_class(cls_name, pretrain_path)` | 클래스 fine-tuning 실행 | 클래스명, pre-train 경로 | fine-tuned 모델 |
| `generate(model, device, prompt)` | 스케치 생성 | 모델, 프롬프트 토큰 | 생성된 토큰 시퀀스 |
| `toks_to_strokes(toks)` | 토큰 → 폴리라인 | 토큰 리스트 | 폴리라인 좌표 리스트 |
| `draw(polylines, ax)` | 폴리라인 시각화 | 폴리라인, matplotlib ax | (이미지 그림) |
| `show_raw_samples(n)` | QuickDraw 원본 표시 | n=4 | (시각화) |
| `show_generated(cls_models, cls_datasets)` | Original vs Generated 비교 | 모델 dict, 데이터셋 dict | (시각화) |
| `save_sequential_strokes(cls_models, cls_datasets)` | Sequential stroke 이미지 저장 | 모델 dict, 데이터셋 dict | (이미지 파일 저장) |
| `main(skip_eda, skip_pretrain, skip_finetune)` | 전체 파이프라인 실행 | boolean flags | (모든 단계 실행 & 파일 저장) |

---

## 참고 자료

- **논문**: [SketchGPT: Sketch Transformer for Semantic Drawing](https://arxiv.org/abs/2405.03099)
- **데이터셋**: [Google QuickDraw Dataset](https://quickdraw.withgoogle.com/data)
- **트랜스포머**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)

---

## 라이선스 및 저작권

- 논문: SketchGPT 저자들에 의해 발표됨
- 구현: 실습/교육 목적 (2026년)
- QuickDraw: Google Creative Commons 라이선스

---

## 주의사항

⚠️ **GPU 메모리**: RTX 3090 (24GB)에 최적화됨. 더 작은 GPU는 배치 크기 감소 필요.

⚠️ **재현성**: 
- `SEED=42`로 고정
- 첫 실행 시 QuickDraw 데이터 자동 다운로드 (클래스당 ~500MB)
- 인터넷 연결 필요

⚠️ **체크포인트 일관성**: EDA 커스터마이징 시 기존 `checkpoints/` 삭제 후 재실행 권장.

---
