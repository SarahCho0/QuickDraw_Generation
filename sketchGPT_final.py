"""
SketchGPT — RTX 3090 로컬 실행 버전
논문: https://arxiv.org/abs/2405.03099

논문 3.4절 구현:
  Pre-training  : 전체 클래스 혼합, 클래스 구별 없음 → pt_best.pt 1개
  Fine-tuning   : 클래스별 개별 fine-tune → gen_<class>.pt 10개
  생성          : 각 클래스 전용 모델로 생성 → 섞임 없음

설치:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install numpy matplotlib tqdm requests scikit-learn Pillow scipy

실행:
  python sketchgpt_final.py

단계별:
  main()                                          # 처음부터 전부
  main(skip_eda=True)                             # EDA 재사용
  main(skip_eda=True, skip_pretrain=True)         # 클래스별 fine-tune만
  main(skip_eda=True, skip_pretrain=True,
       skip_finetune=True)                        # 시각화만 (가중치 로드)

체크포인트 구조:
  checkpoints/
    eda_params.json
    pt_best.pt                  ← 공통 pre-train (1개)
    gen_airplane.pt             ← airplane 전용 fine-tune
    gen_bus.pt
    gen_canoe.pt
    ... (클래스별 10개)
"""

import os, io, math, json, random, requests, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw as PILDraw, ImageFont
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────
# 2. 장치 설정 — GPU 1
# ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

assert torch.cuda.is_available(), "CUDA not found. Please check: nvidia-smi"

GPU_ID = 1   # GPU 0: 점유 중 → GPU 1 사용
DEVICE = torch.device(f"cuda:{GPU_ID}")
torch.cuda.set_device(GPU_ID)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

gpu_name = torch.cuda.get_device_name(GPU_ID)
vram_gb  = torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**3
free_gb  = torch.cuda.mem_get_info(GPU_ID)[0] / 1024**3
print(f"✅ GPU {GPU_ID}: {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  Free: {free_gb:.1f} GB")
if free_gb < 8:
    print(f"⚠️  Low VRAM: {free_gb:.1f} GB")

USE_AMP = True
scaler  = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ─────────────────────────────────────────────────────────
# 3. 경로 설정
# ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "quickdraw_data")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")

RUN_TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", RUN_TS)
SEQ_DIR    = os.path.join(OUTPUT_DIR, "sequential")

for d in (DATA_DIR, CKPT_DIR, OUTPUT_DIR, SEQ_DIR):
    os.makedirs(d, exist_ok=True)

EDA_PATH      = os.path.join(CKPT_DIR, "eda_params.json")
PRETRAIN_PATH = os.path.join(CKPT_DIR, "pt_best.pt")

def finetune_path(cls_name: str) -> str:
    """클래스별 fine-tune 가중치 경로."""
    return os.path.join(CKPT_DIR, f"gen_{cls_name.replace(' ','_')}.pt")

print(f"  Output dir: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────
# 4. 하이퍼파라미터
# ─────────────────────────────────────────────────────────
CLASSES = [
    "airplane", "bus", "canoe", "car", "helicopter",
    "hot air balloon", "motorbike", "sailboat", "submarine", "train"
]
N_TRAIN_PER_CLASS    = 5000
N_VAL_PER_CLASS      = 2000
N_TEST_PER_CLASS     = 2000
N_PRETRAIN_PER_CLASS = 5000

N_PRIMITIVES   = 16
PRIM_LENGTH    = 0.01
TOKEN_BOS = 0; TOKEN_EOS = 1; TOKEN_SEP = 2; TOKEN_PAD = 3
SPECIAL_TOKENS = 4
VOCAB_SIZE     = SPECIAL_TOKENS + N_PRIMITIVES

N_LAYERS = 8; N_HEADS = 8; D_MODEL = 512
D_FF     = D_MODEL * 4
MAX_SEQ  = 256; DROPOUT = 0.1
MAX_SEQ_HARD_LIMIT = 512

PRETRAIN_BATCH      = 64
PRETRAIN_EPOCHS     = 15
PRETRAIN_LR         = 1e-3
GRAD_CLIP           = 1.0
EARLY_STOP_PATIENCE = 3
FINETUNE_GEN_EPOCHS = 10
FINETUNE_GEN_LR     = 1e-4

TEMPERATURE    = 1.0
TOP_K          = 10
PROMPT_RATIO   = 0.5   # 논문: 클래스별 모델이므로 프롬프트 없이도 동작하지만
                       # completion task를 위해 실제 앞부분을 프롬프트로 제공
MIN_NEW_TOKENS = 20

# ─────────────────────────────────────────────────────────
# 5. DataLoader
# ─────────────────────────────────────────────────────────
def make_loader(dataset, batch_size, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=drop_last,
        persistent_workers=True,
    )

# ─────────────────────────────────────────────────────────
# 6. 데이터 전처리
# ─────────────────────────────────────────────────────────
def download_ndjson(class_name: str, max_n: int,
                    recognized_only: bool = True) -> list:
    url_name  = class_name.replace(" ", "%20")
    url       = (f"https://storage.googleapis.com/quickdraw_dataset"
                 f"/full/simplified/{url_name}.ndjson")
    save_path = os.path.join(DATA_DIR, f"{class_name.replace(' ','_')}.ndjson")

    if not os.path.exists(save_path):
        print(f"  Downloading: {class_name} ...")
        r = requests.get(url, stream=True); r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    else:
        print(f"  Cache: {class_name}")

    sketches = []; total_seen = 0
    with open(save_path) as f:
        for line in f:
            total_seen += 1
            data = json.loads(line)
            if recognized_only and not data.get("recognized", True):
                continue
            sketches.append(data["drawing"])
            if len(sketches) >= max_n: break
    print(f"  Loaded: {len(sketches)} (checked {total_seen})")
    return sketches


def drawing_to_stroke3(drawing: list) -> np.ndarray:
    pts, prev_x, prev_y = [], 0, 0
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            pts.append([xs[i]-prev_x, ys[i]-prev_y,
                        1 if i==len(xs)-1 else 0])
            prev_x, prev_y = xs[i], ys[i]
    return np.array(pts, dtype=np.float32)


def normalize_stroke3(s3: np.ndarray) -> np.ndarray:
    """기존 가중치와 호환: x, y 독립 정규화 방식 유지."""
    abs_xy = np.cumsum(s3[:, :2], axis=0)
    mn, mx = abs_xy.min(0), abs_xy.max(0)
    denom  = np.where(mx-mn < 1e-8, 1.0, mx-mn)
    norm   = (abs_xy - mn) / denom
    delta  = np.diff(norm, axis=0, prepend=norm[:1])
    out    = s3.copy(); out[:, :2] = delta
    return out

# ─────────────────────────────────────────────────────────
# 7. Primitive & 토크나이저
# ─────────────────────────────────────────────────────────
def build_primitives(n: int) -> np.ndarray:
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

PRIMITIVES = build_primitives(N_PRIMITIVES)


def prim_id(dx: float, dy: float) -> int:
    vec = np.array([dx, dy]); n = np.linalg.norm(vec)
    if n < 1e-8: return 0
    return int(np.argmax(PRIMITIVES @ (vec / n)))


def scale_factor(dx: float, dy: float) -> int:
    L = math.sqrt(dx**2 + dy**2)
    if L < 1e-8: return 1
    return max(1, min(8, math.ceil(L / PRIM_LENGTH)))


def tokenize(s3: np.ndarray) -> list:
    tokens = [TOKEN_BOS]
    for i, (dx, dy, lift) in enumerate(s3):
        tok = SPECIAL_TOKENS + prim_id(dx, dy)
        tokens.extend([tok] * scale_factor(dx, dy))
        if lift == 1 and i < len(s3)-1:
            tokens.append(TOKEN_SEP)
    tokens.append(TOKEN_EOS)
    return tokens

# ─────────────────────────────────────────────────────────
# 8. EDA
# ─────────────────────────────────────────────────────────
def run_eda(classes, n_sample=500):
    global PRIMITIVES
    all_angles, all_lengths, all_stroke_lens = [], [], []
    for cls_name in classes:
        drawings = download_ndjson(cls_name, n_sample*3)
        count = 0
        for d in drawings:
            s3 = drawing_to_stroke3(d)
            if len(s3) < 5: continue
            s3 = normalize_stroke3(s3)
            for dx, dy, _ in s3:
                L = math.sqrt(dx**2+dy**2)
                if L > 1e-8:
                    all_angles.append(math.atan2(dy, dx))
                    all_stroke_lens.append(L)
            all_lengths.append(len(tokenize(s3)))
            count += 1
            if count >= n_sample: break

    all_angles      = np.array(all_angles)
    all_lengths     = np.array(all_lengths)
    all_stroke_lens = np.array(all_stroke_lens)

    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    axes[0].hist(np.degrees(all_angles), bins=36, range=(-180,180),
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axhline(len(all_angles)/36, color='red', ls='--',
                    label='Uniform baseline')
    axes[0].set(xlabel='Direction (deg)', ylabel='Count',
                title='Stroke Direction Distribution')
    axes[0].legend(); axes[0].grid(alpha=.3)

    p50_l = np.percentile(all_stroke_lens, 50)
    axes[1].hist(all_stroke_lens, bins=60,
                 range=(0, np.percentile(all_stroke_lens,99)),
                 color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(p50_l, color='red', ls='--', lw=2,
                    label=f'median={p50_l:.4f}')
    axes[1].set(xlabel='Stroke length (normalized)',
                title='Stroke Length Distribution')
    axes[1].legend(); axes[1].grid(alpha=.3)

    p95 = np.percentile(all_lengths, 95)
    axes[2].hist(all_lengths, bins=50, color='mediumpurple',
                 edgecolor='white', alpha=0.8)
    axes[2].axvline(p95, color='red', ls='--', lw=2,
                    label=f'95th={p95:.0f}')
    axes[2].set(xlabel='Token length', title='Token Length Distribution')
    axes[2].legend(); axes[2].grid(alpha=.3)

    plt.suptitle(f"EDA: {', '.join(classes)}")
    plt.tight_layout()
    _savefig("eda_analysis.png")

    rec_n_prim   = N_PRIMITIVES
    rec_prim_len = float(p50_l)
    raw_max      = int(p95)
    rec_max_seq  = min(2**math.ceil(math.log2(max(raw_max,64))),
                       MAX_SEQ_HARD_LIMIT)
    print(f"  N_PRIMITIVES={rec_n_prim}  PRIM_LENGTH={rec_prim_len:.5f}  "
          f"MAX_SEQ={rec_max_seq}")
    return rec_n_prim, rec_max_seq, rec_prim_len

# ─────────────────────────────────────────────────────────
# 9. Dataset
# ─────────────────────────────────────────────────────────
class SketchDataset(Dataset):
    def __init__(self, tokens_list, labels, max_seq):
        self.items = []
        for toks, lbl in zip(tokens_list, labels):
            toks = list(toks[:max_seq-1])
            if TOKEN_EOS not in toks: toks.append(TOKEN_EOS)
            toks += [TOKEN_PAD]*(max_seq-len(toks))
            self.items.append((torch.tensor(toks, dtype=torch.long), int(lbl)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


class PartialSketchDataset(Dataset):
    """논문 3.4절: partially drawn sketches로 fine-tune."""
    def __init__(self, tokens_list, labels, max_seq,
                 min_ratio=0.1, max_ratio=0.9):
        self.max_seq = max_seq
        self.min_ratio = min_ratio; self.max_ratio = max_ratio
        self.raw = []
        for toks, lbl in zip(tokens_list, labels):
            toks = list(toks[:max_seq-1])
            if TOKEN_EOS not in toks: toks.append(TOKEN_EOS)
            eos_idx = toks.index(TOKEN_EOS)
            self.raw.append((toks[:eos_idx+1], int(lbl)))
    def __len__(self): return len(self.raw)
    def __getitem__(self, i):
        toks, lbl = self.raw[i]; n = len(toks)
        if n < 4:
            cut = n
        else:
            min_cut = max(2, int(n*self.min_ratio))
            max_cut = max(min_cut+1, int(n*self.max_ratio))
            cut = random.randint(min_cut, max_cut)
        partial = toks[:cut]
        if TOKEN_EOS not in partial: partial.append(TOKEN_EOS)
        partial += [TOKEN_PAD]*(self.max_seq-len(partial))
        return torch.tensor(partial[:self.max_seq], dtype=torch.long), lbl


def build_datasets(classes, n_train, n_val, n_test, n_pretrain, max_seq):
    """Pre-training용: 전체 클래스 혼합 데이터셋."""
    per_cls = []
    for cls_name in classes:
        needed   = n_train+n_val+n_test
        drawings = download_ndjson(cls_name, needed*3)
        cls_toks = []
        for d in drawings:
            s3 = drawing_to_stroke3(d)
            if len(s3) < 5: continue
            s3 = normalize_stroke3(s3); toks = tokenize(s3)
            if len(toks) < 5: continue
            cls_toks.append(toks)
            if len(cls_toks) >= needed: break
        if len(cls_toks) < needed:
            print(f"  ⚠️  {cls_name}: {len(cls_toks)} (target {needed})")
        per_cls.append(cls_toks)

    tr_toks, tr_labs, va_toks, va_labs, te_toks, te_labs = [],[],[],[],[],[]
    cls_counts = []
    for ci, cls_toks in enumerate(per_cls):
        random.shuffle(cls_toks); n = len(cls_toks)
        t1 = min(n_train, int(n*0.5)); t2 = min(n_val, int(n*0.25))
        tr_toks.extend(cls_toks[:t1]);             tr_labs.extend([ci]*t1)
        va_toks.extend(cls_toks[t1:t1+t2]);       va_labs.extend([ci]*t2)
        te_toks.extend(cls_toks[t1+t2:t1+t2*2]); te_labs.extend([ci]*t2)
        cls_counts.append(t1)
        print(f"  {classes[ci]}: train={t1}, val={t2}, test={t2}")

    cls_w = [1.0/c for c in cls_counts]
    def shuf(a,b):
        c=list(zip(a,b)); random.shuffle(c)
        return zip(*c) if c else ([],[])

    tr_t,tr_l = shuf(tr_toks,tr_labs)
    va_t,va_l = shuf(va_toks,va_labs)
    te_t,te_l = shuf(te_toks,te_labs)
    tr_t2,tr_l2 = list(tr_t),list(tr_l)
    sw = [cls_w[l] for l in tr_l2]

    train_ds   = SketchDataset(tr_t2, tr_l2, max_seq)
    val_ds     = SketchDataset(list(va_t), list(va_l), max_seq)
    test_ds    = SketchDataset(list(te_t), list(te_l), max_seq)
    print(f"\n  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_ds, val_ds, test_ds, sw


def build_class_dataset(cls_name: str, n_train: int, n_val: int,
                        max_seq: int):
    """
    논문 3.4절: 클래스별 fine-tune용 단일 클래스 데이터셋.
    pre-train과 달리 해당 클래스 데이터만 사용.
    """
    needed   = n_train + n_val
    drawings = download_ndjson(cls_name, needed*3)
    cls_toks = []
    for d in drawings:
        s3 = drawing_to_stroke3(d)
        if len(s3) < 5: continue
        s3 = normalize_stroke3(s3); toks = tokenize(s3)
        if len(toks) < 5: continue
        cls_toks.append(toks)
        if len(cls_toks) >= needed: break

    if len(cls_toks) < needed:
        print(f"  ⚠️  {cls_name}: {len(cls_toks)} (target {needed})")

    random.shuffle(cls_toks)
    n  = len(cls_toks)
    t1 = min(n_train, int(n*0.67))
    t2 = n - t1

    # label=0 고정 (단일 클래스이므로 label 무의미)
    tr_toks = cls_toks[:t1];  tr_labs = [0]*t1
    va_toks = cls_toks[t1:];  va_labs = [0]*t2

    partial_tr = PartialSketchDataset(tr_toks, tr_labs, max_seq)
    partial_va = PartialSketchDataset(va_toks, va_labs, max_seq)
    # 시각화용 full sequence dataset
    full_tr    = SketchDataset(tr_toks, tr_labs, max_seq)

    print(f"  {cls_name}: ft_train={t1}, ft_val={t2}")
    return partial_tr, partial_va, full_tr

# ─────────────────────────────────────────────────────────
# 10. 모델
# ─────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq, dropout):
        super().__init__()
        self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model,bias=False)
        self.proj=nn.Linear(d_model,d_model,bias=False)
        self.attn_drop=nn.Dropout(dropout)
        self.resid_drop=nn.Dropout(dropout)
        mask=torch.tril(torch.ones(max_seq,max_seq))
        self.register_buffer("causal_mask",mask.view(1,1,max_seq,max_seq))
    def forward(self,x):
        B,L,D=x.shape
        q,k,v=self.qkv(x).split(D,dim=2)
        def h(t): return t.view(B,L,self.n_heads,self.d_head).transpose(1,2)
        q,k,v=h(q),h(k),h(v)
        s=(q@k.transpose(-2,-1))/math.sqrt(self.d_head)
        s=s.masked_fill(self.causal_mask[:,:,:L,:L]==0,float('-inf'))
        w=self.attn_drop(torch.softmax(s,dim=-1))
        return self.resid_drop(self.proj(
            (w@v).transpose(1,2).contiguous().view(B,L,D)))


class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,max_seq,dropout):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=CausalSelfAttention(d_model,n_heads,max_seq,dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.mlp=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),
                               nn.Linear(d_ff,d_model),nn.Dropout(dropout))
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x


class SketchGPT(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,n_layers,d_ff,max_seq,dropout):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(max_seq,d_model)
        self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([
            TransformerBlock(d_model,n_heads,d_ff,max_seq,dropout)
            for _ in range(n_layers)])
        self.ln_f=nn.LayerNorm(d_model)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
        self.apply(self._init_weights)
        n_p=sum(p.numel() for p in self.parameters())
        print(f"  [SketchGPT] L={n_layers}/A={n_heads}/H={d_model} "
              f"params={n_p:,} ({n_p/1e6:.1f}M)")
    def _init_weights(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self,tokens):
        B,L=tokens.shape
        pos=torch.arange(L,device=tokens.device).unsqueeze(0)
        x=self.drop(self.tok_emb(tokens)+self.pos_emb(pos))
        for blk in self.blocks: x=blk(x)
        return self.lm_head(self.ln_f(x))


def make_model():
    """전역 파라미터로 SketchGPT 인스턴스 생성."""
    return SketchGPT(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq=MAX_SEQ, dropout=DROPOUT
    ).to(DEVICE)

# ─────────────────────────────────────────────────────────
# 11. 손실 & 평가
# ─────────────────────────────────────────────────────────
def lm_loss(logits,tokens):
    return F.cross_entropy(
        logits[:,:-1].contiguous().view(-1,logits.size(-1)),
        tokens[:,1:].contiguous().view(-1), ignore_index=TOKEN_PAD)

@torch.no_grad()
def eval_lm(model,loader,device):
    model.eval(); total,n=0.0,0
    for toks,_ in loader:
        toks=toks.to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            total+=lm_loss(model(toks),toks).item()
        n+=1
    return total/max(n,1)

# ─────────────────────────────────────────────────────────
# 12. Pre-training (논문 3.3절)
# 전체 클래스 혼합, 클래스 구별 없는 NTP 사전학습
# ─────────────────────────────────────────────────────────
def pretrain(model, train_ds, val_ds, device, sample_weights=None,
             epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
             batch=PRETRAIN_BATCH, save_path=PRETRAIN_PATH):
    print(f"\n{'='*55}")
    print(f"  PHASE 1: Pre-training  (ep={epochs} lr={lr})")
    print(f"  All {len(CLASSES)} classes mixed, no class discrimination")
    print(f"{'='*55}")
    sampler=None
    if sample_weights is not None:
        sampler=WeightedRandomSampler(
            torch.tensor(sample_weights,dtype=torch.float),
            len(train_ds),replacement=True)
    tr=make_loader(train_ds,batch,sampler=sampler,
                   shuffle=(sampler is None),drop_last=True)
    va=make_loader(val_ds,batch)
    opt=AdamW(model.parameters(),lr=lr,weight_decay=0.01,betas=(0.9,0.95))
    sch=CosineAnnealingLR(opt,T_max=epochs,eta_min=lr*0.1)
    tr_ls,va_ls,best,no_imp=[],[],float('inf'),0
    for ep in range(1,epochs+1):
        model.train(); ep_loss=0.0
        bar=tqdm(tr,desc=f"PT {ep}/{epochs}",ncols=90)
        for toks,_ in bar:
            toks=toks.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss=lm_loss(model(toks),toks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update()
            ep_loss+=loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            lr=f"{sch.get_last_lr()[0]:.5f}")
        sch.step()
        va_loss=eval_lm(model,va,device); tr_loss=ep_loss/len(tr)
        tr_ls.append(tr_loss); va_ls.append(va_loss)
        print(f"  Ep{ep:3d} | train={tr_loss:.4f} val={va_loss:.4f} "
              f"ppl={math.exp(min(va_loss,20)):.1f}")
        if va_loss<best:
            best=va_loss; no_imp=0; torch.save(model.state_dict(),save_path)
        else:
            no_imp+=1
            if no_imp>=EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping (ep={ep})"); break
    _plot_loss(tr_ls,va_ls,"Pre-training Loss","pretrain_loss.png")
    model.load_state_dict(torch.load(save_path,map_location=device,
                                     weights_only=True))
    print(f"✅ Pre-training done  best_val={best:.4f}  saved: {save_path}")
    return model

# ─────────────────────────────────────────────────────────
# 13. Fine-tuning (논문 3.4절)
# 클래스별 개별 fine-tune → 논문: "within a specific class context"
# "multiple models trained on individual classes" (Fig.5 캡션)
# ─────────────────────────────────────────────────────────
def finetune_class(cls_name: str, pretrain_path: str, device,
                   epochs=FINETUNE_GEN_EPOCHS, lr=FINETUNE_GEN_LR,
                   batch=PRETRAIN_BATCH) -> nn.Module:
    """
    논문 3.4절: 클래스 하나에 대한 fine-tuning.
    pre-train 가중치에서 시작하여 해당 클래스 데이터만으로 학습.
    반환: fine-tuned 모델 (메모리에 로드된 상태)
    """
    save_path = finetune_path(cls_name)

    print(f"\n{'='*55}")
    print(f"  PHASE 2: Fine-tuning — {cls_name}")
    print(f"  (ep={epochs} lr={lr}  saved→ {save_path})")
    print(f"{'='*55}")

    # 클래스별 데이터 로드
    partial_tr, partial_va, _ = build_class_dataset(
        cls_name, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, MAX_SEQ)

    # pre-train 가중치에서 시작
    model = make_model()
    model.load_state_dict(torch.load(pretrain_path, map_location=device,
                                     weights_only=True))

    tr  = make_loader(partial_tr, batch, shuffle=True, drop_last=True)
    va  = make_loader(partial_va, batch)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    tr_ls,va_ls,best,no_imp=[],[],float('inf'),0
    for ep in range(1,epochs+1):
        model.train(); ep_loss=0.0
        bar=tqdm(tr,desc=f"FT[{cls_name}] {ep}/{epochs}",ncols=90)
        for toks,_ in bar:
            toks=toks.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss=lm_loss(model(toks),toks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            scaler.step(opt); scaler.update()
            ep_loss+=loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sch.step()
        va_loss=eval_lm(model,va,device); tr_loss=ep_loss/len(tr)
        tr_ls.append(tr_loss); va_ls.append(va_loss)
        print(f"  Ep{ep:3d} | train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss<best:
            best=va_loss; no_imp=0; torch.save(model.state_dict(),save_path)
        else:
            no_imp+=1
            if no_imp>=EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping (ep={ep})"); break

    _plot_loss(tr_ls, va_ls,
               f"Fine-tuning Loss [{cls_name}]",
               f"ft_loss_{cls_name.replace(' ','_')}.png")
    model.load_state_dict(torch.load(save_path,map_location=device,
                                     weights_only=True))
    print(f"✅ Fine-tuning done [{cls_name}]  best_val={best:.4f}")
    return model

# ─────────────────────────────────────────────────────────
# 14. 생성
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, device, prompt=None, max_new=None,
             temperature=TEMPERATURE, top_k=TOP_K,
             min_new_tokens=MIN_NEW_TOKENS):
    if max_new is None: max_new=MAX_SEQ
    model.eval()
    toks=list(prompt) if prompt else [TOKEN_BOS]
    generated=0
    for _ in range(max_new):
        ctx=torch.tensor([toks[-MAX_SEQ:]],dtype=torch.long,device=device)
        logit=model(ctx)[0,-1]
        logit[TOKEN_PAD]=float('-inf')
        if generated<min_new_tokens: logit[TOKEN_EOS]=float('-inf')
        logit/=temperature
        if top_k>0:
            v,_=logit.topk(min(top_k,logit.size(-1)))
            logit[logit<v[-1]]=float('-inf')
        tok=torch.multinomial(F.softmax(logit,-1),1).item()
        toks.append(tok); generated+=1
        if tok==TOKEN_EOS: break
    return toks

# ─────────────────────────────────────────────────────────
# 15. 시각화 유틸
# ─────────────────────────────────────────────────────────
def toks_to_strokes(toks):
    step_size=0.03; polylines=[]; current_pts=[]; x,y=0.5,0.5
    for t in toks:
        if t in (TOKEN_BOS,TOKEN_PAD): continue
        if t==TOKEN_EOS:
            if len(current_pts)>=2: polylines.append(current_pts)
            break
        if t==TOKEN_SEP:
            if len(current_pts)>=2: polylines.append(current_pts)
            current_pts=[(x,y)]; continue
        pid=t-SPECIAL_TOKENS
        if 0<=pid<N_PRIMITIVES:
            d=PRIMITIVES[pid]
            if not current_pts: current_pts.append((x,y))
            x=float(np.clip(x+d[0]*step_size,0.0,1.0))
            y=float(np.clip(y+d[1]*step_size,0.0,1.0))
            current_pts.append((x,y))
    if len(current_pts)>=2: polylines.append(current_pts)
    return polylines


def draw(polylines,ax,title="",color="black",smooth=True):
    if not polylines:
        ax.axis('off'); ax.set_title(title,fontsize=8); return
    for pts in polylines:
        if len(pts)<2: continue
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        if smooth and len(pts)>=4:
            try:
                from scipy.interpolate import splprep,splev
                tck,u=splprep([xs,ys],s=0,k=min(3,len(pts)-1))
                u_new=np.linspace(0,1,len(pts)*5)
                xs_s,ys_s=splev(u_new,tck)
                ax.plot(xs_s,ys_s,color=color,lw=1.5,
                        solid_capstyle='round',solid_joinstyle='round')
            except Exception:
                ax.plot(xs,ys,color=color,lw=1.5,
                        solid_capstyle='round',solid_joinstyle='round')
        else:
            ax.plot(xs,ys,color=color,lw=1.5,
                    solid_capstyle='round',solid_joinstyle='round')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.invert_yaxis(); ax.set_aspect('equal')
    ax.axis('off'); ax.set_title(title,fontsize=8)


def draw_original_quickdraw(drawing,ax,title="",color="black"):
    all_x,all_y=[],[]
    for s in drawing: all_x.extend(s[0]); all_y.extend(s[1])
    if not all_x: return
    xr=max(all_x)-min(all_x) or 1; yr=max(all_y)-min(all_y) or 1; PAD=0.05
    for s in drawing:
        xs=[(x-min(all_x))/xr*(1-2*PAD)+PAD for x in s[0]]
        ys=[(y-min(all_y))/yr*(1-2*PAD)+PAD for y in s[1]]
        ax.plot(xs,ys,color=color,lw=1.5,
                solid_capstyle='round',solid_joinstyle='round')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.invert_yaxis()
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=8)


def _savefig(fname):
    path=os.path.join(OUTPUT_DIR,fname)
    plt.savefig(path,dpi=150,bbox_inches='tight')
    print(f"  [saved] {path}"); plt.close()


def _plot_loss(tr,va,title,fname):
    fig,ax=plt.subplots(figsize=(7,3)); ep=range(1,len(tr)+1)
    ax.plot(ep,tr,'b-o',ms=3,label='Train')
    ax.plot(ep,va,'r-o',ms=3,label='Val')
    ax.set(xlabel='Epoch',ylabel='NLL Loss',title=title)
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); _savefig(fname)


def _polylines_to_pil(polylines, img_size=256, color="black"):
    """폴리라인 → 256×256 PIL. draw()와 동일 파이프라인."""
    dpi=100; sz=img_size/dpi
    fig,ax=plt.subplots(figsize=(sz,sz),dpi=dpi)
    fig.patch.set_facecolor('white')
    draw(polylines,ax,title="",color=color)
    fig.tight_layout(pad=0)
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=dpi,
                bbox_inches='tight',pad_inches=0.02,facecolor='white')
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB").resize(
        (img_size,img_size),Image.LANCZOS)


def _load_font(size=12):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try: return ImageFont.truetype(path, size)
        except Exception: pass
    return ImageFont.load_default()

# ─────────────────────────────────────────────────────────
# 16. 시각화 함수
# ─────────────────────────────────────────────────────────
def show_raw_samples(n=4):
    cmap=matplotlib.cm.get_cmap('tab10',len(CLASSES))
    COLS=[matplotlib.colors.to_hex(cmap(i)) for i in range(len(CLASSES))]
    fig,axes=plt.subplots(len(CLASSES),n,
                          figsize=(n*2.8,len(CLASSES)*2.8))
    for ci,cname in enumerate(CLASSES):
        drawings=download_ndjson(cname,n*5)
        samples=random.sample(drawings,min(n,len(drawings)))
        for j,d in enumerate(samples):
            draw_original_quickdraw(d,axes[ci][j],
                                    title=f"{cname}\n(original)",
                                    color=COLS[ci])
    plt.suptitle("QuickDraw Original Data",y=1.01)
    plt.tight_layout(); _savefig("raw_samples.png")


def show_generated(cls_models: dict, cls_datasets: dict, device, n=4):
    """
    논문 3.4절: 클래스별 전용 모델로 생성.
    cls_models  : {cls_name: model}
    cls_datasets: {cls_name: full SketchDataset (해당 클래스만)}

    original(위) vs generated(아래) 비교 그리드.
    각 클래스 모델이 해당 클래스 데이터만 학습했으므로 섞임 없음.
    """
    cmap=matplotlib.cm.get_cmap('tab10',len(CLASSES))
    COLS=[matplotlib.colors.to_hex(cmap(i)) for i in range(len(CLASSES))]

    fig,axes=plt.subplots(len(CLASSES)*2, n,
                          figsize=(n*2.8, len(CLASSES)*5.6))

    for ci,cname in enumerate(CLASSES):
        model  = cls_models[cname]
        ds     = cls_datasets[cname]
        idxs   = random.sample(range(len(ds)), min(n, len(ds)))

        for j,idx in enumerate(idxs):
            toks_tensor,_ = ds.items[idx]
            full_toks = toks_tensor.tolist()
            real_len  = next((i for i,t in enumerate(full_toks)
                              if t==TOKEN_EOS), len(full_toks))

            # 원본 행
            draw(toks_to_strokes(full_toks), axes[ci*2][j],
                 f"{cname}\n(original)", COLS[ci])

            # 생성 행 — 클래스 전용 모델 사용
            prompt_len = max(5, int(real_len*PROMPT_RATIO))
            prompt     = [t for t in full_toks[:prompt_len]
                          if t not in (TOKEN_EOS, TOKEN_PAD)]
            gen_polylines = toks_to_strokes(
                generate(model, device, prompt=prompt))
            draw(gen_polylines, axes[ci*2+1][j],
                 f"{cname}\n(generated #{j+1})", COLS[ci])

    plt.suptitle(
        f"Original vs Generated  [per-class model, prompt={int(PROMPT_RATIO*100)}%]",
        y=1.01)
    plt.tight_layout()
    _savefig("generated_sketches.png")


def save_sequential_strokes(cls_models: dict, cls_datasets: dict, device,
                             n_per_class=10, img_size=256):
    """
    클래스별 전용 모델로 sequential 저장.
    cls_models  : {cls_name: fine-tuned model}
    cls_datasets: {cls_name: full SketchDataset}

    저장 구조:
      outputs/<ts>/sequential/<class>/sample_XX/
        stroke_001.png … stroke_N.png
        stroke_all.png
        strokes.json / strokes.npy
      sequential/<class>/preview.png
      sequential/overview.png
    """
    print(f"\n[Sequential] {n_per_class}/class × {len(CLASSES)} = "
          f"{n_per_class*len(CLASSES)} total  →  {SEQ_DIR}")

    all_finals = {c:[] for c in CLASSES}
    total = 0

    for ci,cname in enumerate(CLASSES):
        model  = cls_models[cname]
        ds     = cls_datasets[cname]
        idxs   = random.sample(range(len(ds)), min(n_per_class, len(ds)))
        cls_dir= os.path.join(SEQ_DIR, cname.replace(" ","_"))
        os.makedirs(cls_dir, exist_ok=True)

        for si,idx in enumerate(idxs):
            toks_tensor,_ = ds.items[idx]
            full_toks  = toks_tensor.tolist()
            real_len   = next((i for i,t in enumerate(full_toks)
                               if t==TOKEN_EOS), len(full_toks))
            prompt_len = max(5, int(real_len*PROMPT_RATIO))
            prompt     = [t for t in full_toks[:prompt_len]
                          if t not in (TOKEN_EOS, TOKEN_PAD)]
            gen_toks   = generate(model, device, prompt=prompt)
            polylines  = toks_to_strokes(gen_toks)
            if not polylines: continue

            sample_dir = os.path.join(cls_dir, f"sample_{si:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            # 누적 이미지
            for k in range(1, len(polylines)+1):
                _polylines_to_pil(polylines[:k], img_size).save(
                    os.path.join(sample_dir, f"stroke_{k:03d}.png"))

            # 완성본
            final_pil = _polylines_to_pil(polylines, img_size)
            final_pil.save(os.path.join(sample_dir, "stroke_all.png"))
            all_finals[cname].append(final_pil)

            # JSON
            stroke_data=[{"stroke_id":i,
                           "points":[[float(x),float(y)] for x,y in pl]}
                          for i,pl in enumerate(polylines)]
            with open(os.path.join(sample_dir,"strokes.json"),"w") as f:
                json.dump({"class":cname,"sample_id":si,
                           "n_strokes":len(polylines),
                           "img_size":img_size,
                           "prompt_ratio":PROMPT_RATIO,
                           "strokes":stroke_data}, f, indent=2)

            # NPY
            records=[(sid,pid,x,y)
                     for sid,pl in enumerate(polylines)
                     for pid,(x,y) in enumerate(pl)]
            np.save(os.path.join(sample_dir,"strokes.npy"),
                    np.array(records,
                             dtype=[('stroke',np.int32),('point',np.int32),
                                    ('x',np.float32),('y',np.float32)]))
            total+=1
            print(f"  [{total:3d}/{n_per_class*len(CLASSES)}] "
                  f"{cname} sample_{si:02d}: {len(polylines)} strokes")

        if all_finals[cname]:
            _save_class_preview(all_finals[cname], cname, cls_dir, img_size)

    _save_overview(all_finals, CLASSES, SEQ_DIR, img_size, n_per_class)
    print(f"\n[Sequential] Done: {total} samples")
    print(f"  overview  : {SEQ_DIR}/overview.png")
    print(f"  per-class : {SEQ_DIR}/<class>/preview.png")


def _save_class_preview(pil_list, cname, cls_dir, img_size):
    n=len(pil_list); pad=4
    W=n*(img_size+pad)+pad; H=img_size+2*pad+22
    canvas=Image.new("RGB",(W,H),(245,245,245))
    for i,img in enumerate(pil_list):
        canvas.paste(img,(pad+i*(img_size+pad),pad))
    d=PILDraw.Draw(canvas)
    d.text((pad,H-18),cname,fill=(40,40,40),font=_load_font(13))
    path=os.path.join(cls_dir,"preview.png")
    canvas.save(path); print(f"  [preview] {path}")


def _save_overview(all_finals, classes, seq_dir, img_size, n_per_class):
    pad=4; label_w=110; title_h=28
    W=label_w+n_per_class*(img_size+pad)+pad
    H=title_h+len(classes)*(img_size+pad)+pad
    canvas=Image.new("RGB",(W,H),(255,255,255))
    d=PILDraw.Draw(canvas)
    font_ti=_load_font(14); font_sm=_load_font(11)
    d.text((pad,6),
           f"Generated Sketches Overview  "
           f"[per-class model, prompt={int(PROMPT_RATIO*100)}%  "
           f"{n_per_class} samples/class]",
           fill=(20,20,20), font=font_ti)
    for ri,cname in enumerate(classes):
        y0=title_h+ri*(img_size+pad)
        d.text((pad,y0+img_size//2-7), cname, fill=(50,50,50), font=font_sm)
        for ci,img in enumerate(all_finals.get(cname,[])[:n_per_class]):
            canvas.paste(img,(label_w+ci*(img_size+pad),y0))
    path=os.path.join(seq_dir,"overview.png")
    canvas.save(path); print(f"  [overview] {path}")

# ─────────────────────────────────────────────────────────
# 17. main
# ─────────────────────────────────────────────────────────
def main(skip_eda=False, skip_pretrain=False, skip_finetune=False):
    """
    논문 3.3절 + 3.4절 구현.

    skip_finetune=True 일 때:
      checkpoints/gen_<class>.pt 가 모두 존재하면 로드.
      일부만 없으면 없는 클래스만 fine-tune.
    """
    global N_PRIMITIVES, VOCAB_SIZE, MAX_SEQ, PRIM_LENGTH, PRIMITIVES

    print("="*60)
    print(f"  SketchGPT — GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")
    print(f"  Classes : {len(CLASSES)}   Model: L={N_LAYERS}/A={N_HEADS}/H={D_MODEL}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Paper 3.4: per-class fine-tune → {len(CLASSES)} models")
    print("="*60)

    # ── Step 0: EDA ────────────────────────────────────────
    if skip_eda and os.path.exists(EDA_PATH):
        print("\n[Step 0] Load EDA")
        with open(EDA_PATH) as f: eda=json.load(f)
        if eda["n_primitives"]!=N_PRIMITIVES:
            raise ValueError("N_PRIMITIVES mismatch — delete checkpoints/ and rerun")
        N_PRIMITIVES=eda["n_primitives"]; MAX_SEQ=eda["max_seq"]
        PRIM_LENGTH=eda["prim_length"]
        VOCAB_SIZE=SPECIAL_TOKENS+N_PRIMITIVES
        PRIMITIVES=build_primitives(N_PRIMITIVES)
        print(f"  N_PRIMITIVES={N_PRIMITIVES}  MAX_SEQ={MAX_SEQ}  "
              f"PRIM_LENGTH={PRIM_LENGTH:.5f}")
    else:
        print("\n[Step 0] Run EDA")
        rec_n_prim,rec_max_seq,rec_prim_len=run_eda(CLASSES,n_sample=500)
        N_PRIMITIVES=rec_n_prim; VOCAB_SIZE=SPECIAL_TOKENS+N_PRIMITIVES
        MAX_SEQ=min(rec_max_seq,MAX_SEQ_HARD_LIMIT); PRIM_LENGTH=rec_prim_len
        PRIMITIVES=build_primitives(N_PRIMITIVES)
        with open(EDA_PATH,"w") as f:
            json.dump({"n_primitives":N_PRIMITIVES,"max_seq":MAX_SEQ,
                       "prim_length":PRIM_LENGTH,"classes":CLASSES},f,indent=2)

    # ── Step 1: Pre-train용 전체 데이터 ─────────────────────
    print("\n[Step 1] Build datasets (all classes, for pre-training)")
    train_ds,val_ds,test_ds,sw = build_datasets(
        CLASSES, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, N_TEST_PER_CLASS,
        N_PRETRAIN_PER_CLASS, MAX_SEQ)

    # ── Step 2: 모델 초기화 ──────────────────────────────────
    print("\n[Step 2] Init model")
    model = make_model()

    # ── Step 3: Pre-training ────────────────────────────────
    if skip_pretrain and os.path.exists(PRETRAIN_PATH):
        print("\n[Step 3] Load pre-train checkpoint")
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE,
                                         weights_only=True))
        print(f"  loaded: {PRETRAIN_PATH}")
    else:
        print("\n[Step 3] Pre-training (all classes mixed)")
        model = pretrain(model, train_ds, val_ds, DEVICE, sample_weights=sw)

    # ── Step 4: 클래스별 Fine-tuning (논문 3.4절) ────────────
    print(f"\n[Step 4] Per-class fine-tuning  ({len(CLASSES)} classes)")
    cls_models   = {}   # {cls_name: fine-tuned model}
    cls_datasets = {}   # {cls_name: full SketchDataset for visualization}

    for ci, cname in enumerate(CLASSES):
        ft_path = finetune_path(cname)
        print(f"\n  [{ci+1}/{len(CLASSES)}] {cname}")

        if skip_finetune and os.path.exists(ft_path):
            # 기존 가중치 로드
            print(f"  Skip — loading {ft_path}")
            m = make_model()
            m.load_state_dict(torch.load(ft_path, map_location=DEVICE,
                                          weights_only=True))
            # 시각화용 데이터셋 로드 (full sequence)
            _, _, full_tr = build_class_dataset(
                cname, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, MAX_SEQ)
        else:
            # fine-tune 실행
            m = finetune_class(cname, PRETRAIN_PATH, DEVICE)
            _, _, full_tr = build_class_dataset(
                cname, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, MAX_SEQ)

        cls_models[cname]   = m
        cls_datasets[cname] = full_tr

    # ── Step 5-A: Raw QuickDraw ──────────────────────────────
    print("\n[Step 5-A] Raw QuickDraw samples")
    show_raw_samples(n=4)

    # ── Step 5-B: original vs generated 비교 (클래스별 모델) ──
    print("\n[Step 5-B] Generated sketches (per-class model)")
    show_generated(cls_models, cls_datasets, DEVICE, n=4)

    # ── Step 5-C: Sequential 100개 ───────────────────────────
    print("\n[Step 5-C] Sequential stroke images (per-class model)")
    save_sequential_strokes(cls_models, cls_datasets, DEVICE,
                            n_per_class=10, img_size=256)

    print(f"\n✅ Done!  All outputs → {OUTPUT_DIR}")
    print(f"  generated_sketches.png         : original vs generated (per-class)")
    print(f"  sequential/overview.png        : 100개 전체 한눈에 보기")
    print(f"  sequential/<class>/preview.png : 클래스별 10개")
    print(f"  sequential/<class>/sample_XX/  : stroke 누적 이미지 + JSON/NPY")
    print(f"\n  checkpoints/")
    print(f"    pt_best.pt                   : 공통 pre-train")
    for cname in CLASSES:
        print(f"    gen_{cname.replace(' ','_')}.pt")


if __name__ == "__main__":
    # pre-train 가중치(pt_best.pt) 재사용, 클래스별 fine-tune 새로 실행
    # fine-tune이 이미 완료된 클래스는 자동으로 스킵(가중치 로드)
    main(skip_eda=True, skip_pretrain=True, skip_finetune=False)