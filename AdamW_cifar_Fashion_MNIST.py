# -*- coding: utf-8 -*-
# Fashion-MNIST + Simple CNN (NumPy-only) + AdamW (vectorized conv/backprop)
# Карти динамічних режимів: beta2 (X) × beta1 (Y) для кожного класу (0..9).
# Колір точки: X (RMS кроки оновлення по фільтрах) -> D[i]=X[i+1]^2+X[i]^2 і |D[14]-D[k]|<e.
# Без TensorFlow/PyTorch — лише NumPy. Автозавантаження IDX або зовнішні масиви.

from pathlib import Path
from datetime import datetime
import argparse
import os, gzip, urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ===================== КОНФІГ / CLI =====================
def build_cli():
    p = argparse.ArgumentParser(description="Dyn-maps for Fashion-MNIST (NumPy-only CNN + AdamW)")
    # джерело даних
    p.add_argument("--use-external", action="store_true",
                   help="Використати зовнішні масиви з --x, --y або з .npz")
    p.add_argument("--x", type=str, default=None, help="Шлях до X.npy або .npz (ключ x)")
    p.add_argument("--y", type=str, default=None, help="Шлях до y.npy або .npz (ключ y)")
    p.add_argument("--npz", type=str, default=None, help="Шлях до .npz з ключами x,y")
    p.add_argument("--use-train", action="store_true",
                   help="Для автозавантаження IDX: використати train (інакше test).")

    # мережа / навчання
    p.add_argument("--filters", type=int, default=28)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--n-per-class", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs-per-point", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="λ у AdamW (decoupled L2)")
    p.add_argument("--include-wd-in-x", action="store_true",
                   help="Враховувати внесок weight decay у векторі X (за замовчуванням True)",
                   default=True)

    # ґрід
    p.add_argument("--b1-start", type=float, default=0.0)
    p.add_argument("--b1-stop",  type=float, default=1.0)
    p.add_argument("--b1-step",  type=float, default=0.005)
    p.add_argument("--b2-start", type=float, default=0.0)
    p.add_argument("--b2-stop",  type=float, default=1.0)
    p.add_argument("--b2-step",  type=float, default=0.005)

    # кольорова логіка
    p.add_argument("--e-threshold", type=float, default=0.1)
    p.add_argument("--cutoff-x-abs", type=float, default=150.0)

    # вивід
    p.add_argument("--outdir", type=str, default="runs_fmnist_cnn_dynmaps_adamw")
    p.add_argument("--progress", action="store_true", help="Друкувати прогрес по ґріду")
    return p.parse_args()

# ===================== ДОПОМІЖНІ ФУНКЦІЇ =====================
NUM_CLASSES       = 10
IN_H, IN_W, IN_C  = 28, 28, 1          # Fashion-MNIST: grayscale

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def to_float01(x):
    x = x.astype(np.float32)
    x /= 255.0 if x.max() > 1.0 else 1.0
    return x

def ensure_NCHW(x):
    if x.ndim == 3:
        N, H, W = x.shape
        x = x.reshape(N, 1, H, W)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = np.transpose(x, (0, 3, 1, 2))
    elif x.ndim == 4 and x.shape[1] == 1:
        pass
    else:
        raise ValueError("Очікується (N,28,28) або (N,28,28,1) або (N,1,28,28)")
    return x

def build_binary_task(j, X_by_class, N_pos):
    X_pos = X_by_class[j][:N_pos]
    y_pos = np.ones((len(X_pos), 1), dtype=np.float32)
    others = [i for i in range(len(X_by_class)) if i != j]
    need = len(X_pos)
    pool = []
    for k in others:
        take = min(need, len(X_by_class[k]))
        if take > 0:
            pool.append(X_by_class[k][:take])
            need -= take
            if need <= 0:
                break
    if len(pool) == 0:
        raise RuntimeError("Недостатньо негативних прикладів.")
    X_neg = np.concatenate(pool, axis=0).astype(np.float32)
    y_neg = np.zeros((len(X_neg), 1), dtype=np.float32)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ===================== ЗАВАНТАЖУВАЧ FASHION-MNIST (IDX) =====================
FMNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

def _download_if_needed(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[DL] {url} -> {path}")
        urllib.request.urlretrieve(url, path)

def _read_idx_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"Wrong magic for images: {magic}")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    return arr.reshape(n, rows, cols)

def _read_idx_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise ValueError(f"Wrong magic for labels: {magic}")
    n = int.from_bytes(data[4:8], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    return arr.reshape(n,)

def load_fashion_mnist(root="data_fmnist", use_train=True):
    """Повертає (x, y) з IDX-файлів. x: (N,28,28) uint8; y: (N,) uint8."""
    os.makedirs(root, exist_ok=True)
    paths = {
        "train_images": os.path.join(root, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(root, "train-labels-idx1-ubyte.gz"),
        "test_images":  os.path.join(root, "t10k-images-idx3-ubyte.gz"),
        "test_labels":  os.path.join(root, "t10k-labels-idx1-ubyte.gz"),
    }
    for k, url in FMNIST_URLS.items():
        _download_if_needed(url, paths[k])

    if use_train:
        x = _read_idx_images(paths["train_images"])
        y = _read_idx_labels(paths["train_labels"])
    else:
        x = _read_idx_images(paths["test_images"])
        y = _read_idx_labels(paths["test_labels"])
    return x, y

def load_external_arrays(args):
    """Повертає (x, y) з файлів: --npz або --x/--y."""
    if args.npz:
        d = np.load(args.npz)
        if "x" not in d or "y" not in d:
            raise RuntimeError(f"{args.npz} не містить ключів x,y")
        return d["x"], d["y"]
    if args.x is None or args.y is None:
        raise RuntimeError("Для --use-external потрібно вказати --npz або обидва --x і --y")
    if args.x.endswith(".npz"):
        d = np.load(args.x)
        X = d["x"]
    else:
        X = np.load(args.x)
    if args.y.endswith(".npz"):
        d = np.load(args.y)
        y = d["y"]
    else:
        y = np.load(args.y)
    return X, y

# ===================== ВЕКТОРИЗОВАНИЙ SimpleCNN =====================
class SimpleCNN:
    """
    Conv2d (same padding) with F filters of size KxK, ReLU,
    Global Average Pooling (per filter), Dense(F->1), Sigmoid.
    Увесь conv forward/backward векторизовано (im2col + tensordot/einsum).
    """
    def __init__(self, in_c, in_h, in_w, num_filters=32, kernel_size=3, seed=1):
        rng = np.random.default_rng(seed)
        self.F = num_filters
        self.K = kernel_size
        self.in_c = in_c
        self.in_h = in_h
        self.in_w = in_w
        # Conv weights: (F, C, K, K), biases: (F,)
        self.Wc = rng.normal(0, 0.05, size=(self.F, in_c, self.K, self.K)).astype(np.float32)
        self.bc = np.zeros((self.F,), dtype=np.float32)
        # Dense weights: (F, 1), bias: (1,)
        self.Wd = rng.normal(0, 0.05, size=(self.F, 1)).astype(np.float32)
        self.bd = np.zeros((1,), dtype=np.float32)

    @staticmethod
    def _im2col_same(x, K):
        """
        x: (N,C,H,W) -> patches: (N, H, W, C, K, K)
        same padding із падом p=K//2
        """
        N, C, H, W = x.shape
        p = K // 2
        xp = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        # Страйди
        sN, sC, sH, sW = xp.strides
        out_shape = (N, H, W, C, K, K)
        out_strides = (sN, sH, sW, sC, sH, sW)
        patches = np.lib.stride_tricks.as_strided(xp, shape=out_shape, strides=out_strides, writeable=False)
        return patches  # (N,H,W,C,K,K)

    def forward(self, x):
        """
        x: (N, C, H, W)
        Повертає:
          y: (N,1)
          cache: усе, що потрібно для бекварду
        """
        N, C, H, W = x.shape
        patches = self._im2col_same(x, self.K)             # (N,H,W,C,K,K)

        # Conv (векторизовано): (N,H,W,C,K,K) · (F,C,K,K) -> (N,H,W,F)
        conv = np.tensordot(patches, self.Wc, axes=([3,4,5], [1,2,3]))  # (N,H,W,F)
        conv += self.bc[None, None, None, :]                            # +bias

        # ReLU
        relu_mask = (conv > 0.0)
        relu_out = conv * relu_mask                                     # (N,H,W,F)

        # GAP
        gap = np.mean(relu_out, axis=(1,2))                              # (N,F)

        # Dense
        z = gap @ self.Wd + self.bd                                      # (N,1)

        # Sigmoid
        y = 1.0 / (1.0 + np.exp(-z))                                     # (N,1)

        cache = (x, patches, conv, relu_mask, relu_out, gap, z, y)
        return y, cache

    def backward(self, cache, y_true):
        """
        Повертає dWc, dbc, dWd, dbd, mse, dL_dconv, gap
        """
        x, patches, conv, relu_mask, relu_out, gap, z, y_pred = cache
        N = y_pred.shape[0]
        mse = float(np.mean((y_pred - y_true) ** 2))

        # dL/dy (MSE + sigmoid)
        dL_dy = (y_pred - y_true) * (y_pred * (1.0 - y_pred))            # (N,1)

        # Dense grads
        dWd = gap.T @ dL_dy                                              # (F,1)
        dbd = np.sum(dL_dy, axis=0)                                      # (1,)

        # Back to GAP output
        dL_dgap = dL_dy @ self.Wd.T                                      # (N,F)

        # Back to ReLU maps before GAP (均分 по H*W)
        H, W, F = conv.shape[1], conv.shape[2], conv.shape[3]
        dL_drelu = (dL_dgap[:, None, None, :] / (H * W)) * np.ones_like(relu_out)  # (N,H,W,F)

        # ReLU back
        dL_dconv = dL_drelu * relu_mask                                  # (N,H,W,F)

        # Conv grads:
        dbc = np.sum(dL_dconv, axis=(0,1,2))                             # (F,)
        dWc = np.einsum('nhwcij,nhwf->fcij', patches, dL_dconv, optimize=True)  # (F,C,K,K)

        return dWc, dbc, dWd, dbd, mse, dL_dconv, gap

# ===================== AdamW з поверненням ФАКТИЧНИХ ОНОВЛЕНЬ =====================
class AdamWState:
    def __init__(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

def adamw_step_return_updates(params, grads, state: AdamWState, *,
                              lr, beta1, beta2, weight_decay=0.0, eps=1e-8,
                              include_wd_in_return=True, wd_mask=None):
    """
    Decoupled AdamW.
    - params, grads: списки тензорів однакових форм.
    - weight_decay застосовується лише до тих параметрів, де wd_mask[i] == True (наприклад, до ваг, але не bias).
    Повертає список 'steps' тієї ж форми, що params:
      фактичні корекції (що відняли від параметрів), з урахуванням weight decay
      якщо include_wd_in_return=True (для обчислення X).
    """
    if wd_mask is None:
        wd_mask = [True] * len(params)  # за замовчуванням застосовувати всюди (але нижче вимкнемо для bias вручну, якщо треба)
    state.t += 1
    steps = []
    for i, (P, g) in enumerate(zip(params, grads)):
        g = np.asarray(g, dtype=P.dtype)
        if g.shape != P.shape:
            if g.size == P.size:
                g = g.reshape(P.shape)
            else:
                g = np.broadcast_to(g, P.shape)

        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * (g ** 2)
        m_hat = state.m[i] / (1.0 - beta1 ** state.t) if beta1 < 1.0 else state.m[i]
        v_hat = state.v[i] / (1.0 - beta2 ** state.t) if beta2 < 1.0 else state.v[i]

        adaptive_step = lr * (m_hat / (np.sqrt(v_hat) + eps))
        if wd_mask[i] and weight_decay != 0.0:
            wd_step = lr * weight_decay * P
        else:
            wd_step = 0.0

        P -= adaptive_step
        if isinstance(wd_step, (int, float)) and wd_step == 0.0:
            pass
        else:
            P -= wd_step

        step_return = adaptive_step + (wd_step if include_wd_in_return else 0.0)
        if isinstance(step_return, (int, float)) and step_return == 0.0:
            step_return = np.zeros_like(P)
        steps.append(np.array(step_return, dtype=P.dtype).copy())
    return steps

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
def train_point_get_X(
    X, y, * ,
    epochs, batch_size, alpha, beta1, beta2,
    num_filters, kernel_size, seed=1,
    weight_decay=0.01, include_wd_in_x=True
):
    """
    Тренує SimpleCNN 1-vs-rest і повертає:
      - mse_final,
      - X_vec: довжини = num_filters.
        X_vec[j] = RMS( ||ΔWc_f||^2 + Δbc_f^2 )^{1/2} по кроках — ФАКТИЧНІ кроки AdamW для фільтра f.
        (ΔWc_f включає внесок weight decay, якщо include_wd_in_x=True)
    """
    model = SimpleCNN(IN_C, IN_H, IN_W, num_filters=num_filters, kernel_size=kernel_size, seed=seed)
    params = [model.Wc, model.bc, model.Wd, model.bd]
    wd_mask = [True, False, True, False]
    opt = AdamWState(params)

    # Акумуляція RMS по фільтрах (тільки для Conv шару)
    X_sum_sq = np.zeros((num_filters,), dtype=np.float64)
    X_steps  = 0

    N = len(X)
    for ep in range(1, epochs+1):
        idx = np.random.permutation(N)
        Xb, yb = X[idx], y[idx]
        for i in range(0, N, batch_size):
            xb = Xb[i:i+batch_size]
            ytrue = yb[i:i+batch_size]

            ypred, cache = model.forward(xb)
            dWc, dbc, dWd, dbd, mse, _, _ = model.backward(cache, ytrue)

            # нормування на розмір батчу
            bs = max(1, len(xb))
            grads = [dWc/bs, dbc/bs, dWd/bs, dbd/bs]

            steps = adamw_step_return_updates(
                params, grads, opt,
                lr=alpha, beta1=beta1, beta2=beta2,
                weight_decay=weight_decay, include_wd_in_return=include_wd_in_x,
                wd_mask=wd_mask
            )
            step_Wc, step_bc, step_Wd, step_bd = steps

            filt_norms = np.sqrt(np.sum(step_Wc**2, axis=(1,2,3)) + (step_bc**2))
            X_sum_sq += filt_norms**2
            X_steps  += 1

    # RMS по кроках
    X_vec = np.sqrt(X_sum_sq / max(1, X_steps))

    # фінальний MSE на всьому X
    ypred, cache = model.forward(X)
    _, _, _, _, mse_final, _, _ = model.backward(cache, y)
    return mse_final, X_vec

# ===================== ОСНОВНА ФУНКЦІЯ =====================
def main():
    args = build_cli()

    RUNS_DIR = Path(args.outdir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Завантаження входів ----
    if args.use_external:
        x, y = load_external_arrays(args)
    else:
        x, y = load_fashion_mnist(root="data_fmnist", use_train=args.use_train)

    print(f"[DATA] x:{x.shape} y:{y.shape} | source={'external' if args.use_external else ('train' if args.use_train else 'test')}")

    # Підготовка формату/типів
    x = to_float01(x)
    # Визначаємо IN_C глобально з даних
    global IN_C
    IN_C = 1
    x = ensure_NCHW(x)         # (N,1,28,28)
    y = y.astype(np.int32).reshape(-1)

    if not (np.min(y) >= 0 and np.max(y) < NUM_CLASSES):
        raise ValueError("Очікуються мітки 0..9 для Fashion-MNIST.")

    # Розкласти по класах
    X_by_class = []
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0][:args.n_per_class]
        X_by_class.append(x[idx].astype(np.float32))

    beta1_vals = np.arange(args.b1_start, args.b1_stop, args.b1_step)
    beta2_vals = np.arange(args.b2_start, args.b2_stop, args.b2_step)

    total_pts = len(beta1_vals) * len(beta2_vals) * NUM_CLASSES
    done = 0

    for cls in range(NUM_CLASSES):
        if len(X_by_class[cls]) == 0:
            print(f"[skip] class {cls}: немає вибірки")
            continue

        print(f"[CLASS {cls}] building map ...")
        Xj, yj = build_binary_task(cls, X_by_class, args.n_per_class)

        px, py, pc = [], [], []
        for b1 in beta1_vals:
            for b2 in beta2_vals:
                _, X_vec = train_point_get_X(
                    Xj, yj,
                    epochs=args.epochs_per_point, batch_size=args.batch_size,
                    alpha=args.alpha, beta1=b1, beta2=b2,
                    num_filters=args.filters, kernel_size=args.kernel, seed=1,
                    weight_decay=args.weight_decay, include_wd_in_x=args.include_wd_in_x
                )
                color = color_by_D(
                    X_vec,
                    e=args.e_threshold,
                    cutoff=args.cutoff_x_abs,
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
                done += 1
                if args.progress and done % 5 == 0:
                    print(f"  progress: {done}/{total_pts}")

        # Візуалізація + CSV
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(px, py, marker='s', s=16, c=pc)
        plt.title(f"Fashion-MNIST SimpleCNN dynamic regimes (AdamW, class {cls})\n"
                  f"alpha={args.alpha}, wd={args.weight_decay}, epochs/pt={args.epochs_per_point}, filters={args.filters}")
        plt.xlabel(r'$\beta_2$'); plt.ylabel(r'$\beta_1$')
        plt.tight_layout()
        ts = timestamp()
        out_png = RUNS_DIR / f"fmnist_cnn_adamw_map_b1_vs_b2_class{cls}_a{args.alpha}_wd{args.weight_decay}_{ts}.png"
        plt.savefig(out_png, dpi=160); plt.close()
        print(f"[SAVED] {out_png}")

        out_csv = RUNS_DIR / f"fmnist_cnn_adamw_map_b1_vs_b2_class{cls}_a{args.alpha}_wd{args.weight_decay}_{ts}.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("beta1,beta2,color\n")
            for k in range(len(px)):
                f.write(f"{py[k]},{px[k]},{pc[k]}\n")
        print(f"[SAVED] {out_csv}")

    print("[DONE] maps saved to:", RUNS_DIR)

# ===================== ЗАПУСК =====================
if __name__ == "__main__":
    main()
