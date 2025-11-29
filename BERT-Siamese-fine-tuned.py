import os, random, time, itertools
import numpy as np, pandas as pd
from tqdm import tqdm
import joblib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb

# -------------------------
# A100 SPECIFIC OPTIMIZATIONS
# -------------------------
# Enable TensorFloat-32 (TF32) for Ampere GPUs - Significant speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------
# Config & Hyperparameter Grid
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
ACCUM_STEPS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "clean.csv"
OUT_DIR = "hc_outputs_final"; os.makedirs(OUT_DIR, exist_ok=True)

VERBOSE = True
PERFORM_GRID_SEARCH = True 

# --- HYPERPARAMETER GRID ---
BATCH_SIZE_CANDIDATES = [128]
LR_CANDIDATES = [2e-5, 5e-6]
PROJ_DIM_CANDIDATES = [256]
DROPOUT_CANDIDATES = [0.1, 0.3]
NUM_EPOCHS = 10


LOSS_TYPE_CANDIDATES = ['focal', 'margin', 'hybrid']


# Focal
GAMMA_CANDIDATES = [2.0]
ALPHA_CANDIDATES = [0.75]
# Margin
MARGIN_CANDIDATES = [0.3, 0.5]
# Hybrid Weights
FOCAL_WEIGHT_DEFAULT = 1.0
MARGIN_WEIGHT_DEFAULT = 0.3
# -------------------------

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)
df['label'] = df['label'].astype(str).str.strip().str.lower()
df_binary = df[df['label'].isin(['no fit','good fit'])].copy()
df_binary['label_class'] = df_binary['label'].map({'no fit':0,'good fit':1})

if 'job_id' not in df_binary.columns:
    job_to_id = {}
    job_ids = []
    for jd in df_binary['jd_text'].astype(str):
        if jd not in job_to_id: job_to_id[jd] = len(job_to_id)
        job_ids.append(job_to_id[jd])
    df_binary['job_id'] = job_ids

all_jobs = df_binary['job_id'].unique()
train_jobs, temp_jobs = train_test_split(all_jobs, test_size=0.2, random_state=SEED)
val_jobs, test_jobs = train_test_split(temp_jobs, test_size=0.5, random_state=SEED)
train_df = df_binary[df_binary['job_id'].isin(train_jobs)].copy()
val_df   = df_binary[df_binary['job_id'].isin(val_jobs)].copy()
test_df  = df_binary[df_binary['job_id'].isin(test_jobs)].copy()

train_class_counts = train_df['label_class'].value_counts().sort_index()
SCALE_POS_WEIGHT = train_class_counts[0]/train_class_counts[1] if 1 in train_class_counts else 1.0

print(f"\n{'='*70}")
print(f"Dataset Statistics:")
print(f"{'='*70}")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train class distribution: {dict(train_class_counts)}")
print(f"Scale pos weight: {SCALE_POS_WEIGHT:.2f}")
print(f"{'='*70}\n")

# -------------------------
# Dataset 
# -------------------------
class ResumeJobDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        # Tokenize all at once 
        self.jd_enc = tokenizer(df['jd_text'].astype(str).tolist(),
                                truncation=True, padding='max_length',
                                max_length=max_len, return_tensors='pt')
        self.res_enc = tokenizer(df['resume_text'].astype(str).tolist(),
                                 truncation=True, padding='max_length',
                                 max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor(df['label_class'].values,dtype=torch.float)
        self.job_ids = torch.tensor(df['job_id'].values,dtype=torch.long)

    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        return {'input_ids_a':self.jd_enc['input_ids'][idx],
                'attention_mask_a':self.jd_enc['attention_mask'][idx],
                'input_ids_b':self.res_enc['input_ids'][idx],
                'attention_mask_b':self.res_enc['attention_mask'][idx],
                'label_class':self.labels[idx],
                'job_id':self.job_ids[idx]}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ResumeJobDataset(train_df, tokenizer)
val_dataset   = ResumeJobDataset(val_df, tokenizer)
test_dataset  = ResumeJobDataset(test_df, tokenizer)

# -------------------------
# Model 
# -------------------------
class SiameseEncoder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, proj_dim=256, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, proj_dim))
        self.dropout = nn.Dropout(dropout)

    def forward_once(self, input_ids, attention_mask, normalize=True):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_embs = out.last_hidden_state

        # Vectorized mean pooling with attention mask
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embs * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        pooled = self.dropout(pooled)
        emb = self.projection_head(pooled)
        return F.normalize(emb,p=2,dim=1) if normalize else emb

# -------------------------
# Loss Functions
# -------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, job_embs, resume_embs, targets):
        cosine_sim = F.cosine_similarity(job_embs, resume_embs, dim=1)
        probs = (cosine_sim + 1) / 2.0
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        bce_loss = - (targets * torch.log(probs) * self.alpha +
                     (1 - targets) * torch.log(1 - probs) * (1 - self.alpha))
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class CorrectedMarginLoss(nn.Module):
    def __init__(self, margin=0.3, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, job_embs, resume_embs, labels):
        sims = F.cosine_similarity(job_embs, resume_embs, dim=1)
        pos_loss = (1 - sims) * (labels == 1).float() * self.pos_weight
        neg_loss = F.relu(sims - self.margin) * (labels == 0).float() * self.neg_weight
        return (pos_loss.sum() + neg_loss.sum()) / len(labels)

class HybridFocalMarginLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, margin=0.3,
                 focal_weight=1.0, margin_weight=0.3):
        super().__init__()
        self.focal_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.margin_loss = CorrectedMarginLoss(margin=margin)
        self.focal_weight = focal_weight
        self.margin_weight = margin_weight

    def forward(self, job_embs, resume_embs, labels):
        focal = self.focal_loss(job_embs, resume_embs, labels)
        margin = self.margin_loss(job_embs, resume_embs, labels)
        total_loss = self.focal_weight * focal + self.margin_weight * margin
        return total_loss, focal.item(), margin.item()

# -------------------------
# Validation
# -------------------------
@torch.no_grad()
def quick_val_f1(model, val_loader, device=DEVICE):
    model.eval()
    sims_list, labels_list = [], []

    for b in val_loader:
        j = model.forward_once(b['input_ids_a'].to(device, non_blocking=True),
                               b['attention_mask_a'].to(device, non_blocking=True))
        r = model.forward_once(b['input_ids_b'].to(device, non_blocking=True),
                               b['attention_mask_b'].to(device, non_blocking=True))

        batch_sims = F.cosine_similarity(j, r, dim=1)
        batch_probs = (batch_sims + 1) / 2.0

        sims_list.append(batch_probs.cpu())
        labels_list.append(b['label_class'])

    sims = torch.cat(sims_list).numpy()
    labels = torch.cat(labels_list).numpy()

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 17):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t

    return best_f1, best_t

# -------------------------
# Training Function
# -------------------------
def train_siamese_encoder(train_loader, val_loader, loss_type, loss_params,
                          model_config, run_id=None):
    model = SiameseEncoder(
        model_name=MODEL_NAME,
        proj_dim=model_config['proj_dim'],
        dropout=model_config['dropout']
    ).to(DEVICE)

    try:
        model = torch.compile(model)
    except:
        pass

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=model_config['lr'],
        weight_decay=model_config['weight_decay']
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

    # Loss Init
    if loss_type == 'focal':
        loss_fn = BinaryFocalLoss(alpha=loss_params.get('alpha', 0.75), gamma=loss_params.get('gamma', 2.0))
    elif loss_type == 'margin':
        loss_fn = CorrectedMarginLoss(margin=loss_params.get('margin', 0.3))
    elif loss_type == 'hybrid':
        loss_fn = HybridFocalMarginLoss(alpha=loss_params.get('alpha', 0.75), gamma=loss_params.get('gamma', 2.0),
                                        margin=loss_params.get('margin', 0.3))

    scaler = torch.amp.GradScaler('cuda')

    best_val_f1 = -1.0
    best_state = None
    best_threshold = 0.5

    PATIENCE_LIMIT = 4
    patience_counter = 0

    for ep in range(NUM_EPOCHS):
        model.train()
        tot_loss = 0; n_batches = 0
        opt.zero_grad()

        pbar = tqdm(train_loader, desc=f"Run {run_id} | Ep {ep+1}", leave=False)

        for b in pbar:
            input_ids_a = b['input_ids_a'].to(DEVICE, non_blocking=True)
            attn_a = b['attention_mask_a'].to(DEVICE, non_blocking=True)
            input_ids_b = b['input_ids_b'].to(DEVICE, non_blocking=True)
            attn_b = b['attention_mask_b'].to(DEVICE, non_blocking=True)
            y = b['label_class'].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda'):
                j = model.forward_once(input_ids_a, attn_a)
                r = model.forward_once(input_ids_b, attn_b)

                if loss_type == 'hybrid':
                    loss, _, _ = loss_fn(j, r, y)
                else:
                    loss = loss_fn(j, r, y)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (n_batches + 1) % ACCUM_STEPS == 0 or (n_batches + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            tot_loss += loss.item() * ACCUM_STEPS
            n_batches += 1
            pbar.set_postfix({'loss': f"{tot_loss/n_batches:.4f}"})

        val_f1, val_thr = quick_val_f1(model, val_loader)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if hasattr(model, '_orig_mod'):
                best_state = model._orig_mod.state_dict()
            else:
                best_state = model.state_dict()
            best_threshold = val_thr
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE_LIMIT:
            print(f"    ! Early stopping at Ep {ep+1}")
            break

        sched.step()

    return best_state, best_val_f1, best_threshold

# -------------------------
# Main Pipeline
# -------------------------
def main_pipeline():
    print(f"\n{'='*70}\nSTARTING OPTIMIZED GRID SEARCH (Unfrozen BERT on A100)\n{'='*70}")

    best_global_f1 = -1.0
    best_global_config = None
    best_global_model_state = None
    best_global_threshold = 0.5

    run_id = 0
    grid_configs = []

    if PERFORM_GRID_SEARCH:
        for loss_type in LOSS_TYPE_CANDIDATES:
            current_loss_param_sets = []
            if loss_type == 'focal':
                for g in GAMMA_CANDIDATES:
                    for a in ALPHA_CANDIDATES:
                        current_loss_param_sets.append({'gamma': g, 'alpha': a})
            elif loss_type == 'margin':
                for m in MARGIN_CANDIDATES:
                    current_loss_param_sets.append({'margin': m})
            elif loss_type == 'hybrid':
                for g in GAMMA_CANDIDATES:
                    for m in MARGIN_CANDIDATES:
                        current_loss_param_sets.append({
                            'gamma': g, 'alpha': 0.75, 'margin': m,
                            'focal_weight': FOCAL_WEIGHT_DEFAULT, 'margin_weight': MARGIN_WEIGHT_DEFAULT
                        })

            for loss_params in current_loss_param_sets:
                for batch_size in BATCH_SIZE_CANDIDATES:
                    for lr in LR_CANDIDATES:
                        for proj_dim in PROJ_DIM_CANDIDATES:
                            for dropout in DROPOUT_CANDIDATES:
                                grid_configs.append({
                                    'loss_type': loss_type,
                                    'loss_params': loss_params,
                                    'model_config': {
                                        'lr': lr, 'proj_dim': proj_dim,
                                        'dropout': dropout, 'batch_size': batch_size,
                                        'weight_decay': 5e-4
                                    }
                                })
    else:
        grid_configs.append({
            'loss_type': 'focal',
            'loss_params': {'alpha': 0.75, 'gamma': 2.0},
            'model_config': {'lr': 2e-5, 'proj_dim': 256, 'dropout': 0.3, 'batch_size': 64, 'weight_decay': 5e-4}
        })

    print(f"Total configurations to test: {len(grid_configs)}")

    for config in grid_configs:
        run_id += 1
        l_type = config['loss_type']
        l_params = config['loss_params']
        m_conf = config['model_config']

        print(f"[Run {run_id}] Loss:{l_type} | LR:{m_conf['lr']} | BS:{m_conf['batch_size']}")

        train_loader = DataLoader(train_dataset, batch_size=m_conf['batch_size'],
                                  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=m_conf['batch_size'],
                                shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        model_state, val_f1, val_thr = train_siamese_encoder(
            train_loader, val_loader, l_type, l_params, m_conf, run_id=run_id
        )

        print(f"   -> Val F1: {val_f1:.4f}")

        if val_f1 > best_global_f1:
            best_global_f1 = val_f1
            best_global_config = config
            best_global_model_state = model_state
            best_global_threshold = val_thr
            print("   -> ★ NEW BEST ★")

    print(f"\nBest Global F1: {best_global_f1:.4f}")
    print(f"Best Config: {best_global_config}")

    # -------------------------
    # Final Evaluation & Export
    # -------------------------
    print("\nExtracting Embeddings (Best Model)...")
    final_conf = best_global_config['model_config']

    # Initialize Standard Model with Best Config
    model = SiameseEncoder(model_name=MODEL_NAME, proj_dim=final_conf['proj_dim'], dropout=final_conf['dropout']).to(DEVICE)
    model.load_state_dict(best_global_model_state)

    def extract(loader):
        model.eval()
        jd_l, res_l, y_l = [], [], []
        with torch.no_grad():
            for b in tqdm(loader):
                input_ids_a = b['input_ids_a'].to(DEVICE, non_blocking=True)
                attn_a = b['attention_mask_a'].to(DEVICE, non_blocking=True)
                input_ids_b = b['input_ids_b'].to(DEVICE, non_blocking=True)
                attn_b = b['attention_mask_b'].to(DEVICE, non_blocking=True)

                j = model.forward_once(input_ids_a, attn_a, normalize=False).cpu().numpy()
                r = model.forward_once(input_ids_b, attn_b, normalize=False).cpu().numpy()

                jd_l.append(j); res_l.append(r); y_l.append(b['label_class'].numpy())
        return np.vstack(jd_l), np.vstack(res_l), np.concatenate(y_l)

    bs = final_conf['batch_size']
    jd_tr, res_tr, y_tr = extract(DataLoader(train_dataset, batch_size=bs, num_workers=4, pin_memory=True))
    jd_va, res_va, y_va = extract(DataLoader(val_dataset, batch_size=bs, num_workers=4, pin_memory=True))
    jd_te, res_te, y_te = extract(DataLoader(test_dataset, batch_size=bs, num_workers=4, pin_memory=True))

    def create_features(j, r): return np.concatenate([j, r, np.abs(j-r), j*r], axis=1)
    X_train, X_val, X_test = create_features(jd_tr, res_tr), create_features(jd_va, res_va), create_features(jd_te, res_te)

    np.savez_compressed(os.path.join(OUT_DIR, "final_best_features.npz"),
        X_train=X_train, y_train=y_tr, X_val=X_val, y_val=y_va, X_test=X_test, y_test=y_te,
        best_config=best_global_config)
    torch.save(best_global_model_state, os.path.join(OUT_DIR, "best_siamese_model.pt"))

    print("\nTraining Final XGBoost...")
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1,
                            scale_pos_weight=SCALE_POS_WEIGHT)
    clf.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=VERBOSE)
    joblib.dump(clf, os.path.join(OUT_DIR, "best_xgb.joblib"))

    # Optimize XGB Threshold
    probs_val = clf.predict_proba(X_val)[:, 1]
    best_f1_xgb = 0; best_thr_xgb = 0.5
    for t in np.linspace(0.3, 0.8, 11):
        p = (probs_val >= t).astype(int)
        s = f1_score(y_va, p)
        if s > best_f1_xgb: best_f1_xgb, best_thr_xgb = s, t

    # Final Test
    p_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= best_thr_xgb).astype(int)

    print(f"\n[FINAL RESULTS - XGBoost]")
    print(f"F1 Score:  {f1_score(y_te, pred_test):.4f}")
    print(f"Accuracy:  {accuracy_score(y_te, pred_test):.4f}")
    print(f"Precision: {precision_score(y_te, pred_test):.4f}")
    print(f"Recall:    {recall_score(y_te, pred_test):.4f}")

if __name__ == "__main__":
    main_pipeline()

# AI was used for evalution code, debugging, and optimizations