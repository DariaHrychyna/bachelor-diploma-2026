# -*- coding: utf-8 -*-
# Карти динамічних режимів у координатах (beta2 по X, beta1 по Y) для кожного класу CIFAR-10
# NumPy-only MLP + AdamW. Колір точки визначається за схемою X->D та |D[14]-D[k]|<e
# ВАЖЛИВО: X = RMS-величина КОРЕКЦІЇ ВАГ (норма фактичного оновлення) по кожному нейрону
# Де X — величина повної корекції ваг для кожного нейрона (адаптивний крок + weight decay, якщо увімкнено)
import os
import sys
import tarfile
import pickle
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ===================== ПАРАМЕТРИ =====================
HIDDEN_SIZE       = 32         # нейрони в прихованому шарі
NUM_HIDDEN        = 1          # кількість прихованих шарів (>=1)
ACT               = "sigmoid"  # 'sigmoid' або 'tanh'
N_PER_CLASS       = 40         # прикладів на клас (для швидких прогонів)
BATCH_SIZE        = 64

EPOCHS_PER_POINT  = 5          # епох на одну точку карти
ALPHA             = 0.3       # learning rate
WEIGHT_DECAY      = 0.01       # λ у AdamW (decoupled L2)
INCLUDE_WD_IN_X   = True       # якщо True — X рахує і внесок weight decay у фактичному кроці

BETA1_START, BETA1_STOP, BETA1_STEP = 0.0, 1.0, 0.005   # вісь Y
BETA2_START, BETA2_STOP, BETA2_STEP = 0.0, 1.0, 0.005   # вісь X

E_THRESHOLD_E     = 0.1        # поріг e у |D[14]-D[k]|<e
CUTOFF_X_ABS      = 150.0      # якщо |X[ii+1]|>150 -> 'black'

RUNS_DIR          = Path("runs_dynmaps_corr_adamw")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CIFAR_TAR = DATA_DIR / "cifar-10-python.tar.gz"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# ===================== CIFAR-10 =====================
def download_cifar10():
    if CIFAR_TAR.exists():
        return
    print(f"[CIFAR-10] Завантажую {CIFAR_URL} ...")
    try:
        urllib.request.urlretrieve(CIFAR_URL, CIFAR_TAR)
        print(f"[CIFAR-10] Збережено до {CIFAR_TAR}")
    except Exception as e:
        print(f"[CIFAR-10] Не вдалось завантажити: {e}")
        print("Покладіть вручну cifar-10-python.tar.gz у папку 'data/' і перезапустіть.")
        sys.exit(1)

def load_cifar10_numpy():
    download_cifar10()
    with tarfile.open(CIFAR_TAR, "r:gz") as tar:
        members = tar.getmembers()
        train_batches = [m for m in members if "data_batch_" in m.name]
        x_list, y_list = [], []
        for m in sorted(train_batches, key=lambda z: z.name):
            with tar.extractfile(m) as f:
                batch = pickle.load(f, encoding="latin1")
            data = batch["data"]  # (10000, 3072): R(1024), G(1024), B(1024)
            labels = batch.get("labels", batch.get("fine_labels"))
            data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            x_list.append(data)
            y_list.extend(labels)
        x_train = np.concatenate(x_list, axis=0).astype(np.uint8)
        y_train = np.array(y_list, dtype=np.int32)
        return x_train, y_train

def to_gray_flat(x):
    x = x.astype("float32") / 255.0
    R = x[..., 0]; G = x[..., 1]; B = x[..., 2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray.reshape((gray.shape[0], -1))  # (N, 1024)

def build_binary_task(j, X_by_class, N_pos):
    X_pos = X_by_class[j][:N_pos]
    y_pos = np.ones((len(X_pos), 1), dtype=np.float32)
    others = [i for i in range(10) if i != j]
    need = len(X_pos)
    pool = []
    for k in others:
        take = min(need, len(X_by_class[k]))
        if take > 0:
            pool.append(X_by_class[k][:take])
            need -= take
            if need <= 0:
                break
    X_neg = np.vstack(pool).astype(np.float32)
    y_neg = np.zeros((len(X_neg), 1), dtype=np.float32)
    X = np.vstack([X_pos, X_neg])
    y = np.vstack([y_pos, y_neg])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ===================== МЕРЕЖА =====================
def activation(x, kind="sigmoid"):
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    elif kind == "tanh":
        return np.tanh(x)
    else:
        raise ValueError("Unknown activation")

def d_activation(y, kind="sigmoid"):
    if kind == "sigmoid":
        return y * (1.0 - y)
    elif kind == "tanh":
        return 1.0 - y ** 2
    else:
        raise ValueError("Unknown activation")

def init_mlp(input_dim, hidden_size, num_hidden, output_dim=1, seed=1):
    rng = np.random.default_rng(seed)
    Ws, bs = [], []
    prev = input_dim
    for _ in range(num_hidden):
        Ws.append(rng.normal(0, 0.05, size=(prev, hidden_size)).astype(np.float32))
        bs.append(np.zeros((1, hidden_size), dtype=np.float32))
        prev = hidden_size
    Ws.append(rng.normal(0, 0.05, size=(prev, output_dim)).astype(np.float32))
    bs.append(np.zeros((1, output_dim), dtype=np.float32))
    return Ws, bs

def forward_pass(x, Ws, bs, act=ACT):
    layers = [x]  # l0
    preacts = []
    for i in range(len(Ws)-1):
        z = layers[-1] @ Ws[i] + bs[i]
        preacts.append(z)
        layers.append(activation(z, kind=act))
    z = layers[-1] @ Ws[-1] + bs[-1]
    preacts.append(z)
    layers.append(1.0 / (1.0 + np.exp(-z)))
    return layers, preacts

def backward_pass(layers, preacts, y_true, Ws, act=ACT):
    L = len(Ws)
    y_pred = layers[-1]
    mse = float(np.mean((y_pred - y_true) ** 2))

    deltas = [None] * L
    dloss_dy = (y_pred - y_true)
    dy_dz = y_pred * (1.0 - y_pred)
    deltas[-1] = dloss_dy * dy_dz

    for i in range(L-2, -1, -1):
        d_act = d_activation(layers[i+1], kind=act)
        deltas[i] = (deltas[i+1] @ Ws[i+1].T) * d_act

    dWs, dbs = [], []
    for i in range(L):
        dW = layers[i].T @ deltas[i]
        db = np.sum(deltas[i], axis=0, keepdims=True)
        dWs.append(dW)
        dbs.append(db)

    return dWs, dbs, deltas, mse

# ===================== ADAMW (повертає фактичні оновлення) =====================
class AdamWState:
    def __init__(self, Ws, bs):
        self.mW = [np.zeros_like(W) for W in Ws]
        self.vW = [np.zeros_like(W) for W in Ws]
        self.mb = [np.zeros_like(b) for b in bs]
        self.vb = [np.zeros_like(b) for b in bs]
        self.t  = 0

def adamw_step_return_updates(
    Ws, bs, dWs, dbs, st: AdamWState, *,
    lr, beta1, beta2, weight_decay=0.0, eps=1e-8, include_wd_in_return=True):
    """
    Крок AdamW з декупльованим weight decay.
    Повертає:
      - updW[i]: фактична корекція ваг (що було віднято від W), включно з -lr*λ*W якщо include_wd_in_return=True
      - updb[i]: фактична корекція зсувів (без decay)
    """
    st.t += 1
    updW, updb = [], []
    for i in range(len(Ws)):
        # ---- W ----
        st.mW[i] = beta1 * st.mW[i] + (1.0 - beta1) * dWs[i]
        st.vW[i] = beta2 * st.vW[i] + (1.0 - beta2) * (dWs[i] ** 2)

        if beta1 < 1.0:
            mW_hat = st.mW[i] / (1.0 - beta1 ** st.t)
        else:
            mW_hat = st.mW[i]
        if beta2 < 1.0:
            vW_hat = st.vW[i] / (1.0 - beta2 ** st.t)
        else:
            vW_hat = st.vW[i]

        adaptive_step = lr * (mW_hat / (np.sqrt(vW_hat) + eps))
        wd_step       = lr * weight_decay * Ws[i]

        if include_wd_in_return:
            stepW = adaptive_step + wd_step
        else:
            stepW = adaptive_step

        Ws[i] -= adaptive_step
        if weight_decay != 0.0:
            Ws[i] -= wd_step

        updW.append(stepW.copy())

        st.mb[i] = beta1 * st.mb[i] + (1.0 - beta1) * dbs[i]
        st.vb[i] = beta2 * st.vb[i] + (1.0 - beta2) * (dbs[i] ** 2)

        if beta1 < 1.0:
            mb_hat = st.mb[i] / (1.0 - beta1 ** st.t)
        else:
            mb_hat = st.mb[i]
        if beta2 < 1.0:
            vb_hat = st.vb[i] / (1.0 - beta2 ** st.t)
        else:
            vb_hat = st.vb[i]

        stepb = lr * (mb_hat / (np.sqrt(vb_hat) + eps))
        bs[i] -= stepb
        updb.append(stepb.copy())

    return updW, updb

# ===================== КОЛІР =====================
def color_by_D(
    X,
    e,
    cutoff=150.0,
    ref_idx=14,
    compare_idxs=(13,12,11,10,9,8,7,6,5,4,3,2,1,0),
    palette=("red","orange","yellow","green","cyan","blue","violet",
             "#000080","#9370DB","#9932CC","#DDA0DD","#C71585","#191970","#DB7093"),
    default_color="white",
    overflow_color="black",
):
    """
    X — RMS-величина корекції ваг (по нейронах) для обраного шару.
    Якщо будь-який |X[ii+1]|>cutoff -> 'black'.
    D[i] = X[i+1]^2 + X[i]^2.
    Якщо |D[ref_idx]-D[k]|<e -> повертаємо palette[k] (у порядку compare_idxs).
    Інакше -> default_color.
    """
    X = np.asarray(X).ravel()

    if X.size >= 2 and np.any(np.abs(X[1:]) > cutoff):
        return overflow_color

    if X.size < 2:
        return default_color
    D = (X[1:]**2 + X[:-1]**2)

    if not (0 <= ref_idx < D.size):
        return default_color
    Dref = D[ref_idx]

    L = min(len(compare_idxs), len(palette))
    for k in range(L):
        idx = compare_idxs[k]
        if 0 <= idx < D.size and abs(Dref - D[idx]) < e:
            return palette[k]
    return default_color

# ===================== Навчання для однієї точки ґріду =====================
def train_and_get_X_weight_corrections(
    X, y, *,
    input_dim, hidden_size, num_hidden,
    epochs, batch_size, alpha, beta1, beta2,
    layer_to_analyze=1, act=ACT, seed=1,
    weight_decay=WEIGHT_DECAY, include_wd_in_x=INCLUDE_WD_IN_X
):
    """
    Тренуємо MLP з AdamW та повертаємо:
      - mse_final,
      - X_vec: довжина = кількість нейронів на обраному прихованому шарі.
               X_vec[j] = RMS( ||ΔW_col_j||_2 ) по всіх мінібатчах/епохах,
               де ΔW_col_j — ФАКТИЧНИЙ крок оновлення (адаптивний +, за потреби, decay).
    """
    Ws, bs = init_mlp(input_dim, hidden_size, num_hidden, output_dim=1, seed=seed)
    opt = AdamWState(Ws, bs)
    layer_idx = layer_to_analyze - 1
    if not (0 <= layer_idx < num_hidden):
        raise ValueError(f"layer_to_analyze має бути в [1..{num_hidden}]")

    # Акумулятори для RMS по кроках оновлення (per-neuron)
    fan_in = Ws[layer_idx].shape[0]
    X_sum_sq = np.zeros((Ws[layer_idx].shape[1],), dtype=np.float64)
    X_steps  = 0

    N = len(X)
    for ep in range(1, epochs+1):
        idx = np.random.permutation(N)
        Xb, yb = X[idx], y[idx]
        for i in range(0, N, batch_size):
            xb = Xb[i:i+batch_size]
            ytrue = yb[i:i+batch_size]
            layers, preacts = forward_pass(xb, Ws, bs, act=act)
            dWs, dbs, deltas, _ = backward_pass(layers, preacts, ytrue, Ws, act=act)
            # нормування на розмір батча
            dWs = [dW / max(1, len(xb)) for dW in dWs]
            dbs = [db / max(1, len(xb)) for db in dbs]

            updW, updb = adamw_step_return_updates(
                Ws, bs, dWs, dbs, opt,
                lr=alpha, beta1=beta1, beta2=beta2,
                weight_decay=weight_decay, include_wd_in_return=include_wd_in_x
            )

            upd = updW[layer_idx]
            col_norms = np.linalg.norm(upd, axis=0)
            X_sum_sq += (col_norms ** 2)
            X_steps  += 1

    # RMS по кроках
    X_vec = np.sqrt(X_sum_sq / max(1, X_steps))

    # фінальний MSE (на повному train)
    layers_full, preacts_full = forward_pass(X, Ws, bs, act=act)
    _, _, _, mse_full = backward_pass(layers_full, preacts_full, y, Ws, act=act)
    return mse_full, X_vec

# ===================== ОСНОВНА ФУНКЦІЯ =====================
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Дані
    x_tr, y_tr = load_cifar10_numpy()
    x_flat = to_gray_flat(x_tr)

    X_by_class = []
    for cls in range(10):
        idx = np.where(y_tr == cls)[0][:N_PER_CLASS]
        X_by_class.append(x_flat[idx].astype(np.float32))

    input_dim  = x_flat.shape[1]
    beta1_vals = np.arange(BETA1_START, BETA1_STOP, BETA1_STEP)
    beta2_vals = np.arange(BETA2_START, BETA2_STOP, BETA2_STEP)

    for cls in range(10):
        print(f"[CLASS {cls}] building map ...")
        Xj, yj = build_binary_task(cls, X_by_class, N_PER_CLASS)

        px, py, pc = [], [], []
        for b1 in beta1_vals:
            for b2 in beta2_vals:
                mse, X_vec = train_and_get_X_weight_corrections(
                    Xj, yj,
                    input_dim=input_dim, hidden_size=HIDDEN_SIZE, num_hidden=NUM_HIDDEN,
                    epochs=EPOCHS_PER_POINT, batch_size=BATCH_SIZE,
                    alpha=ALPHA, beta1=b1, beta2=b2,
                    layer_to_analyze=1, act=ACT,
                    weight_decay=WEIGHT_DECAY, include_wd_in_x=INCLUDE_WD_IN_X
                )
                color = color_by_D(
                    X_vec,
                    e=E_THRESHOLD_E,
                    cutoff=CUTOFF_X_ABS,
                    ref_idx=14,
                    compare_idxs=(13,12,11,10,9,8,7,6,5,4,3,2,1,0),
                    palette=("red","orange","yellow","green","cyan","blue","violet",
                             "#000080","#9370DB","#9932CC","#DDA0DD","#C71585","#191970","#DB7093"),
                    default_color="white",
                    overflow_color="black",
                )
                px.append(b2)   # X: beta2
                py.append(b1)   # Y: beta1
                pc.append(color)

        # Візуалізація та CSV
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(px, py, marker='s', s=14, c=pc)
        plt.title(f"Dynamic regimes map by weight corrections (AdamW, class {cls})\nalpha={ALPHA}, wd={WEIGHT_DECAY}, epochs/pt={EPOCHS_PER_POINT}")
        plt.xlabel(r'$\beta_2$'); plt.ylabel(r'$\beta_1$')
        plt.tight_layout()
        ts = timestamp()
        out_png = RUNS_DIR / f"map_adamw_corr_b1_vs_b2_class{cls}_a{ALPHA}_wd{WEIGHT_DECAY}_{ts}.png"
        plt.savefig(out_png, dpi=150); plt.close()
        print(f"[SAVED] {out_png}")

        out_csv = RUNS_DIR / f"map_adamw_corr_b1_vs_b2_class{cls}_a{ALPHA}_wd{WEIGHT_DECAY}_{ts}.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("beta1,beta2,color\n")
            for k in range(len(px)):
                f.write(f"{py[k]},{px[k]},{pc[k]}\n")
        print(f"[SAVED] {out_csv}")

    print("[DONE] All class maps are saved to:", RUNS_DIR)


if __name__ == "__main__":
    main()
