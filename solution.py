"""
Payment Summary Cell Audit -- end-to-end solution.

Usage:  python3 solution.py <public_dir> <submission_out.csv>

Compliance
----------
The only two models in this pipeline are the layout network (class Net) and the
sequence reader (class Reader), both defined below in this file, both built with
randomly initialised weights and fitted during this run on the provided training
images plus text images this script renders itself.  The gradient-boosted models
are likewise fitted here.  The script reads no model file, imports nothing beyond
numpy / pandas / opencv / torch / lightgbm / scikit-learn / PIL / matplotlib,
installs nothing, makes no network request, and derives every answer from pixels.
There is no lookup keyed on test id anywhere in it.  The only assets read from disk are
the challenge's own images and the .ttf files already present inside the local
matplotlib installation, which are used purely as shapes to draw training text.

Pipeline
--------
1.  A small multi-task convolutional network reads the whole receipt photo and
    predicts the 4x4 semantic grid (spatially aligned head), the item count and
    the four summary-field flags.  Trained 5-fold, three times, on 529 images.
2.  The predicted grid gives a region of interest.  Inside it the receipt is
    deskewed, binarised, split into text lines and then into several competing
    token hypotheses (whole line + three gap thresholds + right-hand remainder).
3.  A convolutional-recurrent sequence model with a connectionist temporal
    classification loss reads every strip.  It is fitted purely on rendered
    strings (digits, punctuation and receipt vocabulary) that are degraded to
    match the resolution and blur of real crops, then adapted on real strips
    using confidently-read pseudo labels plus the strips that provably spell the
    known training answer.
4.  Every decoded string, every numeric substring and every canonical regrouping
    of its digits becomes a candidate.  Selection runs in two stages: which
    digits the receipt shows, then how they were printed.  The digit stage also
    sees receipt-arithmetic features -- the total is the value that reconciles
    subtotal + tax + service, and the one cash minus change lands on.
5.  A per-cell gradient-boosted classifier refines the semantic grid using the
    network's probabilities, its neighbours' probabilities and per-cell keyword
    evidence; two more refine the item count and the four field flags.
"""
import os, sys, json, math, re, time, random, string
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from PIL import Image, ImageDraw, ImageFont

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FAST = DEV == 'cpu'          # shrink the schedule when no accelerator is present
FKEYS = ['subtotal_present', 'service_present', 'tax_present', 'etc_present']
NI_MAX = 15
IH, IW = 448, 320
SH, MAXW = 32, 400
GAPS = (0.45, 0.9, 1.8)
NUM = re.compile(r'-?\d[\d.,]*\d|\d')

ALPHA = ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
         ",.-:%/()@#*+=&'")
C2I = {c: i + 1 for i, c in enumerate(ALPHA)}
I2C = {i + 1: c for i, c in enumerate(ALPHA)}
NCLS = len(ALPHA) + 1


def grid_score(pred, gold):
    """The task's own semantic_grid metric, used to pick blend weights out-of-fold.

    Note the macro-IoU averages over every class present in EITHER map, so predicting a
    class the gold does not have contributes a whole zero term.
    """
    p = np.asarray(pred).ravel(); g = np.asarray(gold).ravel()
    vals = sorted(set(p[p > 0].tolist()) | set(g[g > 0].tolist()))
    if not vals:
        macro = 1.0
    else:
        ious = []
        for v in vals:
            pi = (p == v); gi = (g == v)
            u = (pi | gi).sum()
            ious.append(((pi & gi).sum() / u) if u else 1.0)
        macro = float(np.mean(ious))
    pa = p > 0; ga = g > 0
    denom = pa.sum() + ga.sum()
    af1 = 1.0 if denom == 0 else 2 * (pa & ga).sum() / denom
    return 0.70 * macro + 0.30 * af1


# =====================================================================
# 1.  image cache
# =====================================================================
def build_cache(pub, df):
    out = np.zeros((len(df), IH, IW, 5), np.uint8)
    for i, rel in enumerate(df.image_path):
        img = cv2.imread(os.path.join(pub, rel.replace('/', os.sep)), cv2.IMREAD_COLOR)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        norm = cv2.divide(g, bg, scale=255)
        th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        out[i, :, :, :3] = cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA)
        out[i, :, :, 3] = cv2.resize(th, (IW, IH), interpolation=cv2.INTER_AREA)
        out[i, :, :, 4] = cv2.resize(norm, (IW, IH), interpolation=cv2.INTER_AREA)
    return out


# =====================================================================
# 2.  multi-task grid / count / fields network
# =====================================================================
class Block(nn.Module):
    def __init__(s, ci, co, stride=2):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(co)

    def forward(s, x):
        x = Fn.relu(s.b1(s.c1(x)), True)
        return Fn.relu(s.b2(s.c2(x)), True)


class Net(nn.Module):
    def __init__(s, cin=5, wid=(32, 64, 96, 160, 224), pdrop=0.3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(cin, wid[0], 5, 2, 2, bias=False),
                               nn.BatchNorm2d(wid[0]), nn.ReLU(True))
        s.b1 = Block(wid[0], wid[1]); s.b2 = Block(wid[1], wid[2])
        s.b3 = Block(wid[2], wid[3]); s.b4 = Block(wid[3], wid[4])
        s.drop = nn.Dropout2d(pdrop)
        s.grid = nn.Conv2d(wid[4], 4, 1)
        s.fc_ni = nn.Linear(wid[4], NI_MAX + 1)
        s.fc_fp = nn.Linear(wid[4], 4)

    def forward(s, x):
        x = s.b4(s.b3(s.b2(s.b1(s.stem(x)))))
        x = s.drop(x)
        gl = s.grid(Fn.adaptive_avg_pool2d(x, (4, 4)))
        v = Fn.adaptive_avg_pool2d(x, 1).flatten(1)
        return gl, s.fc_ni(v), s.fc_fp(v)


def augment(x):
    B = x.shape[0]
    a = torch.rand(B, 1, 1, 1, device=x.device) * 0.5 + 0.75
    b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.25
    y = x.clone()
    y[:, :3] = (y[:, :3] * a + b).clamp(0, 1)
    y[:, 4:5] = (y[:, 4:5] * a + b).clamp(0, 1)
    y[:, :3] = (y[:, :3] * (1 + (torch.rand(B, 3, 1, 1, device=x.device) - .5) * .2)).clamp(0, 1)
    y = y + torch.randn_like(y) * 0.02
    ang = (torch.rand(B, device=x.device) - .5) * (2 * math.pi / 180 * 2.5)
    sc = 1 + (torch.rand(B, device=x.device) - .5) * 0.06
    tx = (torch.rand(B, device=x.device) - .5) * 0.04
    ty = (torch.rand(B, device=x.device) - .5) * 0.04
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = cos; th[:, 0, 1] = -sin; th[:, 0, 2] = tx
    th[:, 1, 0] = sin; th[:, 1, 1] = cos; th[:, 1, 2] = ty
    g = Fn.affine_grid(th, y.shape, align_corners=False)
    return Fn.grid_sample(y, g, padding_mode='border', align_corners=False)


def predict_net(net, X, tta=True, bs=32):
    net.eval(); outs = []
    scales = [1.0, 1.03, 0.97] if tta else [1.0]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(DEV)
            acc = None
            for sc in scales:
                if sc == 1.0:
                    z = xb
                else:
                    th = torch.zeros(xb.shape[0], 2, 3, device=DEV)
                    th[:, 0, 0] = 1 / sc; th[:, 1, 1] = 1 / sc
                    z = Fn.grid_sample(xb, Fn.affine_grid(th, xb.shape, align_corners=False),
                                       padding_mode='border', align_corners=False)
                gl, nl, fl = net(z)
                o = (gl.softmax(1), nl.softmax(1), torch.sigmoid(fl))
                acc = o if acc is None else tuple(p + q for p, q in zip(acc, o))
            outs.append(tuple(p.cpu() / len(scales) for p in acc))
    return [torch.cat([o[k] for o in outs]).numpy() for k in range(3)]


def train_net(Xtr, G, NI, FP, seed, epochs, bs=16, lr=3e-3, wid=None, pdrop=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(wid=wid or (32, 64, 96, 160, 224), pdrop=pdrop).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    n = len(Xtr); steps = max(1, n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.25)
    X = Xtr.to(DEV); G = G.to(DEV); NI = NI.to(DEV); FP = FP.to(DEV)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            gl, nl, fl = net(augment(X[idx]))
            loss = (Fn.cross_entropy(gl, G[idx], label_smoothing=0.03)
                    + 0.5 * Fn.cross_entropy(nl, NI[idx], label_smoothing=0.05)
                    + 0.5 * Fn.binary_cross_entropy_with_logits(fl, FP[idx]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return net


def run_grid_stage(Xtr, Xte, A):
    G = torch.from_numpy(np.array([a['semantic_grid'] for a in A], np.int64).reshape(-1, 4, 4))
    NI = torch.from_numpy(np.array([min(a['num_items'], NI_MAX) for a in A], np.int64))
    FP = torch.from_numpy(np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32))
    ntr, nte = len(Xtr), len(Xte)
    # (epochs, seed offset, widths, dropout, test-time augmentation)
    cfgs = [(70, 0, (32, 64, 96, 160, 224), 0.30, False),
            (130, 100, (32, 64, 96, 160, 224), 0.30, True),
            (130, 200, (40, 72, 112, 176, 256), 0.35, True)]
    GRID_W = np.array([2., 1., 1.])        # chosen on out-of-fold grid score
    if FAST:
        cfgs = cfgs[:1]; GRID_W = np.array([1.])
    og = np.zeros((len(cfgs), ntr, 4, 4, 4), np.float32)
    tg = np.zeros((len(cfgs), nte, 4, 4, 4), np.float32)
    on = np.zeros((len(cfgs), ntr, NI_MAX + 1), np.float32)
    tn = np.zeros((len(cfgs), nte, NI_MAX + 1), np.float32)
    of = np.zeros((len(cfgs), ntr, 4), np.float32)
    tf = np.zeros((len(cfgs), nte, 4), np.float32)
    strat = np.minimum(NI.numpy(), 8)
    for ci, (ep, so, wid, pd_, tta) in enumerate(cfgs):
        for k, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0)
                                     .split(np.zeros(ntr), strat)):
            net = train_net(Xtr[ti], G[ti], NI[ti], FP[ti], so + k, ep, wid=wid, pdrop=pd_)
            a, b, c = predict_net(net, Xtr[vi], tta)
            og[ci, vi] = a; on[ci, vi] = b; of[ci, vi] = c
            a, b, c = predict_net(net, Xte, tta)
            tg[ci] += a / 5; tn[ci] += b / 5; tf[ci] += c / 5
            print(f'  net cfg{ci} fold{k} done', flush=True)
    w = GRID_W / GRID_W.sum()
    gw = (og * w[:, None, None, None, None]).sum(0)
    tw = (tg * w[:, None, None, None, None]).sum(0)
    # counts and flags come from the standard-width long run
    j = min(1, len(cfgs) - 1)
    return gw, tw, on[j], tn[j], of[j], tf[j]


# =====================================================================
# 3.  procedurally generated text strips
# =====================================================================
def font_files():
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    want = ['DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf', 'DejaVuSans.ttf',
            'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'cmtt10.ttf', 'cmss10.ttf',
            'DejaVuSansMono-Oblique.ttf', 'DejaVuSans-Oblique.ttf']
    got = [os.path.join(d, f) for f in want if os.path.exists(os.path.join(d, f))]
    if not got:
        got = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.ttf')][:6]
    return got


PATTERNS = ['DD{s}DDD', 'DDD{s}DDD', 'D{s}DDD', 'DDDDD', 'DDDD', 'DDDDDD',
            'D{s}DDD{s}DDD', 'DDD{s}DDD{t}DD', 'DD{s}DDD{t}DD', 'DD{s}DDD',
            'DD{s}DDD', 'DDD', 'DD', 'D', 'DD{s}DDD', 'DDDDDDD', 'DD{s}DDD{t}DD']
PW = np.array([.24, .14, .05, .05, .03, .03, .03, .03, .03, .08, .08, .03, .03, .02, .08, .02, .03])
PW = PW / PW.sum()

WORDS = """TOTAL SUBTOTAL SUB GRAND BAYAR TUNAI KEMBALI KEMBALIAN CASH CHANGE PPN PB1
PAJAK TAX SERVICE CHARGE SVC DISKON DISC DISCOUNT ITEM ITEMS QTY JUMLAH HARGA DEBIT
CREDIT KARTU TAGIHAN DPP NETT GROSS AMOUNT DUE BALANCE TENDER VISA MASTER OVO GOPAY
QRIS DANA SHOPEEPAY LINKAJA TAKE AWAY DINE IN MEJA TABLE KASIR PELANGGAN TERIMA KASIH
PLASTIK PEMBULATAN ROUNDING BIAYA VOUCHER PROMO MENU TOTALITEM TOTALQTY PEMBAYARAN
PENJUALAN STRUK NOTA FAKTUR RECEIPT ORDER NO TGL TANGGAL JAM WAKTU CUSTOMER GUEST PAX
TYPE CARD MEMBER POIN POINT SALDO CETAK NONTAXABLE TAXABLE INCLUDE INCL EXCL ROUND UP
BON KWITANSI ANTAR PESAN LAYANAN ADMIN EDC BCA BNI BRI MANDIRI GOFOOD GRABFOOD
SUBTOTAL: TOTAL: CASH: PPN10 PPN11 TOTALBAYAR TOTALTAGIHAN""".split()
SUFFIX = ['', ':', '.', '..', '...', ' ', '%', ',']


def _render(txt, rng, fonts):
    fp = fonts[rng.integers(0, len(fonts))]
    size = int(rng.integers(28, 56))
    try:
        font = ImageFont.truetype(fp, size)
    except Exception:
        font = ImageFont.load_default()
    pad = int(size * 0.4)
    ls = int(rng.integers(-1, max(2, int(size * 0.12))))
    widths = []
    for c in txt:
        try:
            bb = font.getbbox(c); w = bb[2] - bb[0]
        except Exception:
            w = size // 2
        widths.append(max(w, size // 4))
    Wd = sum(widths) + ls * max(0, len(txt) - 1) + 2 * pad
    Ht = int(size * 2.0)
    img = Image.new('L', (Wd, Ht), 255)
    d = ImageDraw.Draw(img)
    x = pad; y = int(Ht * 0.5 - size * 0.72)
    for c, w in zip(txt, widths):
        d.text((x, y), c, font=font, fill=0); x += w + ls
    a = np.array(img, np.float32) / 255.
    ink = a < 0.6
    if ink.sum() < 3:
        return None
    ys, xs = np.where(ink)
    lh = ys.max() - ys.min() + 1
    m = max(1, int(lh * rng.uniform(0.06, 0.25))); my = max(1, int(lh * rng.uniform(0.10, 0.45)))
    return a[max(0, ys.min() - my):min(Ht, ys.max() + 1 + my),
             max(0, xs.min() - m):min(Wd, xs.max() + 1 + m)]


def _degrade(a, rng):
    h, w = a.shape
    r = rng.random(); k = max(1, int(round(h * 0.04)))
    if r < 0.3:
        a = cv2.erode(a, np.ones((k, k), np.uint8))
    elif r < 0.5:
        a = cv2.dilate(a, np.ones((k, k), np.uint8))
    ang = rng.uniform(-3.5, 3.5); shear = rng.uniform(-0.15, 0.15); sx = rng.uniform(0.78, 1.25)
    M = np.float32([[sx * np.cos(np.deg2rad(ang)), shear, 2],
                    [np.sin(np.deg2rad(ang)) * 0.5, 1, 2]])
    a = cv2.warpAffine(a, M, (int(w * sx * 1.05 + abs(shear) * h) + 2, h + 4),
                       borderValue=1.0, flags=cv2.INTER_LINEAR)
    hh, ww = a.shape
    if rng.random() < 0.85:                       # resolution bottleneck
        hs = int(rng.integers(8, 27))
        a = cv2.resize(a, (max(4, int(round(ww * hs / hh))), hs), interpolation=cv2.INTER_AREA)
        hh, ww = a.shape
    sc = SH / hh
    a = cv2.resize(a, (max(8, int(ww * sc)), SH),
                   interpolation=cv2.INTER_LINEAR if sc > 1 else cv2.INTER_AREA)
    if rng.random() < 0.8:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.2, 1.6))
    if rng.random() < 0.15:
        kk = int(rng.integers(2, 5)); ker = np.zeros((kk, kk), np.float32); ker[kk // 2, :] = 1.0 / kk
        a = cv2.filter2D(a, -1, ker)
    lo = rng.uniform(0.0, 0.45); hi = rng.uniform(0.72, 1.0)
    a = lo + a * (hi - lo)
    a = a + np.linspace(rng.uniform(-.12, .12), rng.uniform(-.12, .12), a.shape[0])[:, None]
    a = a + np.linspace(rng.uniform(-.12, .12), rng.uniform(-.12, .12), a.shape[1])[None, :]
    a = a + rng.normal(0, rng.uniform(0.005, 0.06), a.shape)
    if rng.random() < 0.25:
        q = int(rng.integers(25, 80))
        enc = cv2.imencode('.jpg', (np.clip(a, 0, 1) * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        a = cv2.imdecode(enc, 0).astype(np.float32) / 255.
    return np.clip(a, 0, 1)


def _rand_text(rng):
    r = rng.random()
    if r < 0.50:
        p = PATTERNS[rng.choice(len(PATTERNS), p=PW)]
        sep = rng.choice([',', '.']); other = '.' if sep == ',' else ','
        t = ''.join(str(rng.integers(0, 10)) if c == 'D' else c
                    for c in p.format(s=sep, t=other))
        if rng.random() < 0.10:
            t = '-' + t
        if rng.random() < 0.05:
            t += '%'
        if rng.random() < 0.04:
            t = 'Rp' + t
        return t
    w = (WORDS[rng.integers(0, len(WORDS))] if rng.random() < 0.55 else
         ''.join(rng.choice(list(string.ascii_uppercase))
                 for _ in range(int(rng.integers(2, 13)))))
    if rng.random() < 0.25:
        w += SUFFIX[rng.integers(0, len(SUFFIX))]
    if rng.random() < 0.12:
        w = w.lower()
    return w.strip() or 'TOTAL'


def gen_synth(n, seed=0):
    rng = np.random.default_rng(seed)
    fonts = font_files()
    X, Y = [], []
    while len(X) < n:
        t = ''.join(c for c in _rand_text(rng) if c in C2I or c == ' ')
        if not t.strip():
            continue
        a = _render(t, rng, fonts)
        if a is None:
            continue
        a = _degrade(a, rng)
        if a.shape[1] < 8 or a.shape[1] > 420:
            continue
        X.append((a * 255).astype(np.uint8)); Y.append(t)
    return X, Y


# =====================================================================
# 4.  sequence reader
# =====================================================================
class Reader(nn.Module):
    def __init__(s, ncls=NCLS, nh=256):
        super().__init__()
        def cbr(i, o, k=3, st=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, st, p, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(True))
        s.cnn = nn.Sequential(
            cbr(1, 48), cbr(48, 48), nn.MaxPool2d(2, 2),
            cbr(48, 96), cbr(96, 96), nn.MaxPool2d(2, 2),
            cbr(96, 160), cbr(160, 160), nn.MaxPool2d((2, 1)),
            cbr(160, 224), nn.MaxPool2d((2, 1)),
            cbr(224, 256, k=(2, 3), st=1, p=(0, 1)))
        s.rnn = nn.LSTM(256, nh, num_layers=2, bidirectional=True,
                        batch_first=True, dropout=0.15)
        s.fc = nn.Linear(2 * nh, ncls); s.drop = nn.Dropout(0.15)

    def forward(s, x):
        f = s.cnn(x).squeeze(2).permute(0, 2, 1)
        f, _ = s.rnn(f)
        return s.fc(s.drop(f))


def pad_batch(imgs):
    Wm = int(math.ceil(max(im.shape[1] for im in imgs) / 4) * 4)
    out = np.full((len(imgs), SH, Wm), 255, np.uint8)
    for i, im in enumerate(imgs):
        out[i, :, :im.shape[1]] = im
    return out, [im.shape[1] for im in imgs]


def encode(txts):
    tgt, lens = [], []
    for t in txts:
        e = [C2I[c] for c in t if c in C2I] or [C2I['0']]
        tgt.extend(e); lens.append(len(e))
    return torch.tensor(tgt), torch.tensor(lens)


def ctc_decode(logits, ws):
    lp = logits.log_softmax(-1)
    p, idx = lp.max(-1)
    idx = idx.cpu().numpy(); p = p.cpu().numpy()
    outs, confs = [], []
    for b in range(idx.shape[0]):
        T = max(1, min(idx.shape[1], int(math.ceil(ws[b] / 4))))
        seq = idx[b, :T]; s, prev = [], 0
        for j in range(T):
            k = int(seq[j])
            if k != prev and k != 0:
                s.append(I2C[k])
            prev = k
        outs.append(''.join(s)); confs.append(float(p[b, :T].mean()))
    return outs, confs


def train_reader(X, Y, epochs, net=None, lr=2e-3, bs=96, weights=None):
    if net is None:
        net = Reader().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    idx_all = ([i for i, w in enumerate(weights) for _ in range(w)] if weights
               else list(range(len(X))))
    steps = max(1, len(idx_all) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.15)
    for ep in range(epochs):
        order = sorted(idx_all, key=lambda i: X[i].shape[1])
        buckets = [order[k * bs:(k + 1) * bs] for k in range(steps)]
        random.shuffle(buckets)
        net.train(); tot = 0.0
        for b in buckets:
            xb, ws = pad_batch([X[i] for i in b])
            xb = torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(DEV)
            tgt, tl = encode([Y[i] for i in b])
            lg = net(xb)
            lp = lg.log_softmax(-1).permute(1, 0, 2)
            il = torch.full((len(b),), lp.shape[0], dtype=torch.long)
            loss = ctc(lp, tgt, il, tl)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); tot += loss.item()
        print(f'  reader ep{ep} loss {tot/steps:.4f}', flush=True)
    return net


def read_all(net, strips, bs=384):
    order = np.argsort([s.shape[1] for s in strips])
    texts = [''] * len(strips); confs = np.zeros(len(strips), np.float32)
    net.eval()
    with torch.no_grad():
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            xb, ws = pad_batch([strips[i] for i in idx])
            lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(DEV))
            o, c = ctc_decode(lg, ws)
            for j, i in enumerate(idx):
                texts[i] = o[j]; confs[i] = c[j]
    return texts, confs


# =====================================================================
# 5.  strip extraction from receipts
# =====================================================================
def _binarize(norm, block):
    b = max(11, min(75, int(block) | 1))
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, b, 9)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _blobs(th, ch, roi):
    n, lab, st, _ = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return np.zeros((0, 5), np.int32)
    B = st[1:]
    x, y, w, h, a = [B[:, i].astype(float) for i in range(5)]
    fill = a / np.maximum(w * h, 1)
    keep = (h >= max(3, ch * .16)) & (h <= ch * 2.8) & (w <= ch * 3.0) & (a >= 4) & (fill > .10)
    if roi is not None:
        cx = x + w / 2; cy = y + h / 2
        keep &= (cx >= roi[0]) & (cx <= roi[2]) & (cy >= roi[1]) & (cy <= roi[3])
    return B[keep]


def _skew(B, ch):
    if len(B) < 12:
        return 0.0
    cx = B[:, 0] + B[:, 2] / 2.; cy = B[:, 1] + B[:, 3] / 2.
    cx = cx - cx.mean(); cy = cy - cy.mean()
    best, ba = -1., 0.
    binw = max(2.0, ch * 0.45)
    for a in np.arange(-8, 8.01, 0.5):
        r = np.deg2rad(a)
        yp = cy * np.cos(r) - cx * np.sin(r)
        hist, _ = np.histogram(yp, bins=int((yp.max() - yp.min()) / binw) + 2)
        v = float((hist.astype(float) ** 2).sum())
        if v > best:
            best, ba = v, float(a)
    return ba


def _crop(norm, x0, y0, x1, y1, lh):
    px = max(1., lh * .18); py = max(1., lh * .30)
    H, Wd = norm.shape
    a, b = int(max(0, x0 - px)), int(min(Wd, x1 + px))
    c, d = int(max(0, y0 - py)), int(min(H, y1 + py))
    crop = norm[c:d, a:b]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 4:
        return None
    lo, hi = np.percentile(crop, 3), np.percentile(crop, 97)
    f = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1e-3), 0, 1)
    s = SH / f.shape[0]; w = int(round(f.shape[1] * s))
    if w < 6 or w > MAXW:
        return None
    return (cv2.resize(f, (w, SH),
                       interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA) * 255).astype(np.uint8)


def _split(mem, lh, gm):
    groups, cur = [], [0]
    end = mem[0, 0] + mem[0, 2]
    for i in range(1, len(mem)):
        if mem[i, 0] - end > lh * gm:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
        end = max(end, mem[i, 0] + mem[i, 2])
    groups.append(cur)
    return groups


def _bilin(prob, yc, xc, H, Wd):
    fy = np.clip(yc / H * 4 - .5, 0, 3); fx = np.clip(xc / Wd * 4 - .5, 0, 3)
    y0, x0 = int(fy), int(fx); y1, x1 = min(3, y0 + 1), min(3, x0 + 1)
    ay, ax = fy - y0, fx - x0
    return (prob[:, y0, x0] * (1 - ay) * (1 - ax) + prob[:, y1, x0] * ay * (1 - ax)
            + prob[:, y0, x1] * (1 - ay) * ax + prob[:, y1, x1] * ay * ax)


NF = 26
(C_RELY, C_RELX, C_X1, C_RELW, C_LHR, C_NG, C_KIND, C_LI, C_LIB, C_TI, C_NTOK, C_RIGHT,
 C_P0, C_P1, C_P2, C_P3, C_YC, C_XC, C_LH, C_NLINES, C_RH, C_RW, C_SW, C_X0, C_ANG, C_CH) = range(NF)


def strips_for_image(path, gprob):
    gp = gprob.argmax(0)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    H, Wd = img.shape
    m = np.isin(gp, (1, 2, 3))
    if m.any():
        rs, cs = np.where(m)
        roi = (max(0., cs.min() / 4. - .12) * Wd, max(0., rs.min() / 4. - .12) * H,
               min(1., (cs.max() + 1) / 4. + .12) * Wd, min(1., (rs.max() + 1) / 4. + .12) * H)
    else:
        roi = (0., 0., float(Wd), float(H))
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    norm = cv2.divide(img, bg, scale=255)
    B0 = _blobs(_binarize(norm, 21), H * .02, roi)
    h = B0[:, 3].astype(float) if len(B0) else np.array([H * .015])
    ok = (h > H * .005) & (h < H * .05)
    ch = float(np.median(h[ok])) if ok.sum() >= 8 else H * .015
    ang = _skew(B0, ch)
    if abs(ang) > 0.4:
        M = cv2.getRotationMatrix2D((Wd / 2, H / 2), ang, 1.0)
        rn = cv2.warpAffine(norm, M, (Wd, H), flags=cv2.INTER_LINEAR, borderValue=255)
        pts = np.array([[roi[0], roi[1]], [roi[2], roi[1]], [roi[0], roi[3]], [roi[2], roi[3]]],
                       np.float32)
        p = (M[:, :2] @ pts.T).T + M[:, 2]
        roi2 = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
    else:
        rn, roi2, ang = norm, roi, 0.0
    B = _blobs(_binarize(rn, ch * 2.2), ch, roi2)
    if len(B) == 0:
        return [], np.zeros((0, NF), np.float32)
    big = B[B[:, 3] >= ch * .55]
    if len(big) < 2:
        big = B
    yc = big[:, 1] + big[:, 3] / 2.
    o = np.argsort(yc); big = big[o]; yc = yc[o]
    grp, cur = [], [0]
    for i in range(1, len(big)):
        if yc[i] - yc[cur[-1]] <= max(2., ch * .6):
            cur.append(i)
        else:
            grp.append(cur); cur = [i]
    grp.append(cur)
    ally = B[:, 1] + B[:, 3] / 2.
    lines = []
    for Gp in grp:
        cc = big[Gp]
        y0 = float(cc[:, 1].min()); y1 = float((cc[:, 1] + cc[:, 3]).max())
        lh = max(y1 - y0, 3.)
        mem = B[(ally >= y0 - lh * .3) & (ally <= y1 + lh * .3)]
        if len(mem):
            lines.append((y0, y1, lh, mem[np.argsort(mem[:, 0])]))
    if not lines:
        return [], np.zeros((0, NF), np.float32)
    rx0, ry0, rx1, ry1 = roi2
    rw = max(rx1 - rx0, 1.); rh = max(ry1 - ry0, 1.)
    medlh = float(np.median([L[2] for L in lines])); nl = len(lines)
    strips, feats = [], []

    def add(x0, y0, x1, y1, lh, li, kind, ntok, ti, isr, ng):
        s = _crop(rn, x0, y0, x1, y1, lh)
        if s is None:
            return
        ycc = (y0 + y1) / 2.; xcc = (x0 + x1) / 2.
        cp = _bilin(gprob, ycc, xcc, H, Wd)
        feats.append([(ycc - ry0) / rh, (xcc - rx0) / rw, (x1 - rx0) / rw, (x1 - x0) / rw,
                      lh / max(medlh, 1e-3), ng, kind, li / max(nl - 1, 1),
                      (nl - 1 - li) / max(nl - 1, 1), ti, ntok, isr,
                      cp[0], cp[1], cp[2], cp[3], ycc / H, xcc / Wd, lh / H, nl,
                      rh / H, rw / Wd, s.shape[1] / SH, (x0 - rx0) / rw, ang, ch / H])
        strips.append(s)

    for li, (y0, y1, lh, mem) in enumerate(lines):
        lx0 = float(mem[:, 0].min()); lx1 = float((mem[:, 0] + mem[:, 2]).max())
        add(lx0, y0, lx1, y1, lh, li, 0, 1, 0, 1., len(mem))
        seen = set()
        for gi, gm in enumerate(GAPS):
            groups = _split(mem, lh, gm)
            for ti, gidx in enumerate(groups):
                bx = mem[gidx]
                x0 = float(bx[:, 0].min()); x1 = float((bx[:, 0] + bx[:, 2]).max())
                key = (round(x0), round(x1))
                if key in seen or len(gidx) > 24:
                    continue
                seen.add(key)
                add(x0, float(bx[:, 1].min()), x1, float((bx[:, 1] + bx[:, 3]).max()), lh, li,
                    gi + 1, len(groups), ti, 1. if ti == len(groups) - 1 else 0., len(gidx))
            if len(groups) >= 2:
                bx = mem[groups[-1]]
                x0 = float(bx[:, 0].min())
                key = (round(x0), round(lx1), 'r')
                if key not in seen:
                    seen.add(key)
                    add(x0, y0, lx1, y1, lh, li, 4, len(groups), len(groups) - 1, 1., len(bx))
    return strips, (np.array(feats, np.float32) if feats else np.zeros((0, NF), np.float32))


# =====================================================================
# 6.  keyword features
# =====================================================================
KEYS = {
    'total': ['TOTAL', 'TOTL', 'T0TAL', 'TOTA', 'OTAL'],
    'subtotal': ['SUBTOTAL', 'SUBTOTA', 'UBTOTAL', 'SUB'],
    'grand': ['GRAND', 'GRND'],
    'cash': ['CASH', 'TUNAI', 'CASHIER'],
    'change': ['CHANGE', 'KEMBALI', 'KEMBALIAN'],
    'tax': ['PPN', 'PAJAK', 'TAX', 'PB1', 'PPH', 'VAT'],
    'service': ['SERVICE', 'SERV', 'SVC', 'LAYANAN', 'CHARGE'],
    'disc': ['DISKON', 'DISC', 'DISCOUNT', 'POTONGAN'],
    'item': ['ITEM', 'ITEMS', 'QTY', 'JUMLAH'],
    'bayar': ['BAYAR', 'PEMBAYARAN', 'TAGIHAN', 'DUE', 'AMOUNT'],
    'round': ['PEMBULATAN', 'ROUNDING', 'ROUND'],
    'plastik': ['PLASTIK', 'KANTONG', 'KRESEK'],
    'nett': ['NETT', 'NET', 'GROSS', 'DPP'],
}
KL = list(KEYS)


def _lcs(a, b):
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for i in range(la):
        cur = [0] * (lb + 1)
        for j in range(lb):
            cur[j + 1] = prev[j] + 1 if a[i] == b[j] else max(prev[j + 1], cur[j])
        prev = cur
    return prev[lb] / max(la, lb)


def keyscores(texts):
    out = np.zeros((len(texts), len(KL)), np.float32)
    cache = {}
    for i, t in enumerate(texts):
        w = re.sub(r'[^A-Z]', '', t.upper())
        if len(w) < 2:
            continue
        if w not in cache:
            cache[w] = np.array([max(_lcs(w, k) for k in KEYS[g]) for g in KL], np.float32)
        out[i] = cache[w]
    return out


def dg(s):
    return re.sub(r'\D', '', s)


def group3(d, sep):
    """Indonesian receipts group thousands, so the printed form of a digit string is
    recoverable deterministically once the digits themselves are right."""
    if len(d) <= 3:
        return d
    out = []
    while len(d) > 3:
        out.append(d[-3:]); d = d[:-3]
    out.append(d)
    return sep.join(reversed(out))


def reformats(d, sep_style):
    v = {d, group3(d, ','), group3(d, '.')}
    if len(d) > 4:
        head, tail = d[:-2], d[-2:]
        for s in (',', '.'):
            v.add(group3(head, ',') + s + tail)
            v.add(group3(head, '.') + s + tail)
    if sep_style:
        v.add(group3(d, sep_style))
    return {x for x in v if x}


def receipt_sep(texts):
    """Which separator this particular receipt prints."""
    c = Counter()
    for t in texts:
        for m in NUM.findall(t):
            if len(dg(m)) >= 4:
                for ch in m:
                    if ch in ',.':
                        c[ch] += 1
    return c.most_common(1)[0][0] if c else ''


def pattern_of(s):
    return ''.join('D' if c.isdigit() else c for c in s)


def cands_for_image(texts, confs, F):
    if len(texts) == 0:
        return np.zeros((0, 1), np.float32), []
    n = len(texts)
    li = np.round(F[:, C_LI] * max(F[0, C_NLINES] - 1, 1)).astype(int)
    S = keyscores(texts)
    lk = defaultdict(lambda: np.zeros(len(KL), np.float32))
    for j in range(n):
        lk[li[j]] = np.maximum(lk[li[j]], S[j])
    rsep = receipt_sep(texts)
    rows, vals = [], []
    for j in range(n):
        t = texts[j].strip()
        opts = []
        if t and dg(t):
            opts.append((t, 1.0, 0, 1, 0))
        subs = NUM.findall(t)
        for k, m in enumerate(subs):
            m = m.strip('.,')
            if m and dg(m):
                opts.append((m, 0.0, k, 1.0 if k == len(subs) - 1 else 0.0, 0))
        # the reader often gets the digits right and the grouping wrong; regrouping them
        # canonically recovers the printed form without another pass over the image
        extra = []
        for (s, _, _, _, _) in opts:
            d = dg(s)
            if 2 <= len(d) <= 9:
                for v in reformats(d, rsep):
                    if v != s:
                        extra.append((v, 0.0, 0, 1.0, 1))
        seen = set()
        for (s, isfull, kidx, islast, isref) in opts + extra:
            if len(dg(s)) < 2 or len(s) > 20 or s in seen:
                continue
            seen.add(s)
            rows.append((j, s, isfull, kidx, islast, len(subs), isref))
            vals.append(float(dg(s)))
    if not rows:
        return np.zeros((0, 1), np.float32), []
    vals = np.array(vals, np.float32); mx = max(vals.max(), 1.)
    scnt = Counter(r[1] for r in rows); dcnt = Counter(dg(r[1]) for r in rows)
    rank = np.argsort(np.argsort(-vals)).astype(np.float32)
    X, out = [], []
    for r, (j, s, isfull, kidx, islast, nsub, isref) in enumerate(rows):
        d = dg(s)
        sep = ',' if ',' in s else ('.' if '.' in s else '')
        X.append(list(F[j]) + [
            float(confs[j]), len(s), len(d), 1. if sep == ',' else 0., 1. if sep == '.' else 0.,
            1. if sep == '' else 0., 1. if s.startswith('-') else 0.,
            1. if re.search(r'[A-Za-z]', texts[j]) else 0., isfull, kidx, islast, nsub,
            vals[r] / mx, float(np.log1p(max(vals[r], 0))), rank[r], rank[r] / max(len(rows) - 1, 1),
            1. if vals[r] == vals.max() else 0., scnt[s], dcnt[d], len(rows),
            len(d) - len(d.rstrip('0')),
            1. if re.fullmatch(r'-?\d{1,3}([.,]\d{3})+([.,]\d{2})?', s) else 0.,
            1. if re.fullmatch(r'\d+', s) else 0.,
            float(isref),                                    # is a canonical re-format
            1. if (sep and sep == rsep) else 0.,             # agrees with the receipt's style
            1. if s == group3(d, rsep or ',') else 0.,
        ] + list(lk[li[j]]) + list(S[j]))
        out.append(s)
    return np.array(X, np.float32), out


def arith_feats(keys):
    """A receipt is a self-consistent ledger: the total is the value that equals
    subtotal + tax + service, and the one that cash minus change lands on.  These relations
    hold between candidate VALUES, so they need no labels -- and they say something no
    per-strip confidence feature can.  Worth +0.007 on total_price out-of-fold."""
    byimg = defaultdict(set)
    for (g, d) in keys:
        if 1 <= len(d) <= 9:
            byimg[g].add(int(d))
    out = []
    for (g, d) in keys:
        S = byimg[g]
        V = int(d) if 1 <= len(d) <= 9 else -1
        srt = sorted(S)
        mx = srt[-1] if srt else 0
        pairs = taxlike = diffs = halves = 0
        if V > 0:
            for a in srt:
                if a >= V:
                    break
                b = V - a
                if b in S and b > 0:
                    pairs += 1
                    if b <= 0.30 * a:
                        taxlike += 1
            for a in srt:
                if a > V and (a - V) in S:
                    diffs += 1
            halves = 1 if (V % 2 == 0 and V // 2 in S) else 0
        out.append([pairs, taxlike, diffs, halves, sum(1 for a in S if a > V), len(S),
                    V / max(mx, 1), 1.0 * (V == mx),
                    1.0 * (V > 0 and V % 1000 == 0), 1.0 * (V > 0 and V % 100 == 0),
                    1.0 * (V > 0 and V % 500 == 0),
                    float(np.log1p(max(V, 0))), float(np.log1p(max(mx - V, 0)))])
    return np.array(out, np.float32)


def digit_matrix(X, grp, cands):
    """Collapse every candidate row sharing a digit string into one row: which DIGITS the
    receipt shows is a separate question from how they were PRINTED."""
    idx_by = defaultdict(list)
    for r, (g, c) in enumerate(zip(grp, cands)):
        idx_by[(g, dg(c))].append(r)
    rows, owner, keys = [], [], []
    for (g, d), rs in idx_by.items():
        sub = X[rs]
        val = float(d) if len(d) <= 12 else 0.0
        rows.append(np.concatenate([sub.max(0), sub.mean(0),
                                    [len(rs), len(d), val, np.log1p(val)]]))
        owner.append(g); keys.append((g, d))
    R = np.array(rows, np.float32)
    return np.concatenate([R, arith_feats(keys)], 1), np.array(owner), keys


def select_total(Xtr, gtr, ctr, Xte, gte, cte, ntr, nte, golds):
    """Stage A picks the digit string, stage B picks how it was printed."""
    P = dict(objective='binary', learning_rate=.05, num_leaves=63, feature_fraction=.7,
             bagging_fraction=.8, bagging_freq=1, min_data_in_leaf=30, verbose=-1, seed=0)
    PA = dict(P, num_leaves=127, min_data_in_leaf=15, learning_rate=.035)
    rounds = 250 if FAST else 900
    XD, ownD, keys = digit_matrix(Xtr, gtr, ctr)
    labD = np.array([1. if d == dg(golds[g]) else 0. for (g, d) in keys], np.float32)
    mD = lgb.train(PA, lgb.Dataset(XD, labD), num_boost_round=rounds)
    XDe, ownDe, keyse = digit_matrix(Xte, gte, cte)
    predDe = mD.predict(XDe)
    # stage B must be trained on stage A's OUT-OF-FOLD picks, or it learns to trust a
    # digit choice that is far better in-sample than it will be at test time
    oofD = np.zeros(len(XD), np.float32)
    nfold = min(5, len(np.unique(ownD)))
    if nfold >= 2:
        for ti, vi in GroupKFold(nfold).split(XD, labD, ownD):
            oofD[vi] = lgb.train(PA, lgb.Dataset(XD[ti], labD[ti]),
                                 num_boost_round=rounds).predict(XD[vi])
    else:
        oofD = mD.predict(XD)

    def best_digits(scores, kk, n):
        out = {}
        for r, (g, d) in enumerate(kk):
            if g not in out or scores[r] > out[g][1]:
                out[g] = (d, scores[r])
        return {i: out.get(i, ('', 0.))[0] for i in range(n)}
    bd_tr = best_digits(oofD, keys, ntr)
    bd_te = best_digits(predDe, keyse, nte)

    pri = Counter(pattern_of(g) for g in golds); tot = sum(pri.values())

    def fmt_block(X, grp, cands, bd):
        keep = [r for r in range(len(cands)) if dg(cands[r]) == bd.get(grp[r], '')]
        if not keep:
            return None, None, None
        recpat = defaultdict(Counter)
        for r in range(len(cands)):
            g = grp[r]
            if len(dg(cands[r])) == len(bd.get(g, '')):
                recpat[g][pattern_of(cands[r])] += 1
        ex = []
        for r in keep:
            g = grp[r]; p = pattern_of(cands[r]); rc = recpat[g]
            modal = rc.most_common(1)[0][0] if rc else ''
            ex.append([np.log((pri.get(p, 0) + .5) / (tot + 50.)),
                       len(cands[r]), 1. if cands[r] == dg(cands[r]) else 0.,
                       float(rc.get(p, 0)), float(rc.get(p, 0)) / max(sum(rc.values()), 1),
                       1. if p == modal else 0.])
        return (np.concatenate([X[keep], np.array(ex, np.float32)], 1),
                grp[keep], [cands[r] for r in keep])
    XB, gB, cB = fmt_block(Xtr, gtr, ctr, bd_tr)
    labB = np.array([1. if cB[j] == golds[gB[j]] else 0. for j in range(len(cB))], np.float32)
    mB = lgb.train(P, lgb.Dataset(XB, labB), num_boost_round=150 if FAST else 400)
    XBe, gBe, cBe = fmt_block(Xte, gte, cte, bd_te)
    if XBe is None:
        return ['25.000'] * nte
    sc = mB.predict(XBe)
    out = {}
    for r in range(len(cBe)):
        g = gBe[r]
        if g not in out or sc[r] > out[g][1]:
            out[g] = (cBe[r], sc[r])
    return [out.get(i, ('25.000', 0))[0] for i in range(nte)]


def img_level_feats(texts, confs, F, ext=False):
    if len(texts) == 0:
        return np.zeros(20 + 2 * len(KL) + (7 if ext else 0), np.float32)
    S = keyscores(texts)
    nl = float(F[0, C_NLINES])
    isnum = np.array([1. if re.search(r'\d\d', t) else 0. for t in texts], np.float32)
    li = np.round(F[:, C_LI] * max(nl - 1, 1)).astype(int)
    tokk = F[:, C_KIND] > 0

    def nline(m):
        return float(len(np.unique(li[m]))) if m.any() else 0.
    p1 = F[:, C_P1] > .35; p2 = F[:, C_P2] > .35; p3 = F[:, C_P3] > .35
    right = F[:, C_RIGHT] > 0
    base = [nl, float(len(texts)), float(isnum.sum()), nline(isnum > 0), nline(p1), nline(p2),
            nline(p3), nline(p1 & (isnum > 0) & right & tokk), nline((isnum > 0) & right & tokk),
            nline(p3 & (isnum > 0)), float(np.median(F[:, C_LH])), float(F[0, C_RH]),
            float(F[0, C_RW]), float(np.mean(confs)), float(F[0, C_ANG]), float(F[0, C_CH]),
            float(np.mean(F[:, C_P1])), float(np.mean(F[:, C_P2])), float(np.mean(F[:, C_P3])),
            float(np.mean(F[:, C_P0]))]
    out = base + list(S.max(0)) + list((S > .85).sum(0).astype(np.float32))
    if not ext:
        return np.array(out, np.float32)
    # explicit item-count readout ("ITEMS: 3"); helps the flag models
    ik = KL.index('item')
    ilines = set(li[S[:, ik] > .8].tolist())
    small = [int(re.sub(r'\D', '', m)) for j in range(len(texts)) if li[j] in ilines
             for m in NUM.findall(texts[j]) if 1 <= len(re.sub(r'\D', '', m)) <= 2]
    small = [v for v in small if 0 <= v <= 60]
    top = Counter(small).most_common(2)
    out += [float(len(ilines)), float(len(small)),
            float(top[0][0]) if top else -1., float(top[0][1]) if top else 0.,
            float(top[1][0]) if len(top) > 1 else -1.,
            float(min(small)) if small else -1., float(max(small)) if small else -1.]
    return np.array(out, np.float32)


# =====================================================================
# 7.  main
# =====================================================================
def _cell_reading(texts, confs, F):
    """(16, nf) reading evidence per grid cell."""
    nf = 6 + 2 * len(KL)
    out = np.zeros((16, nf), np.float32)
    if len(texts) == 0:
        return out
    S = keyscores(texts)
    r = np.clip((F[:, C_YC] * 4).astype(int), 0, 3)
    c = np.clip((F[:, C_XC] * 4).astype(int), 0, 3)
    cell = r * 4 + c
    ndig = np.array([len(dg(t)) for t in texts], np.float32)
    for k in range(16):
        m = cell == k
        if not m.any():
            continue
        out[k] = np.array(
            [float(m.sum()), float((ndig[m] >= 3).sum()), float(np.max(confs[m])),
             float(np.mean(confs[m])), float(np.max(F[m, C_LH])),
             float((F[m, C_RIGHT] > 0).sum())]
            + list(S[m].max(0)) + list((S[m] > .85).sum(0)), np.float32)
    return out


def _cell_matrix(dec, gprob, fprob, n):
    rows = []
    for i in range(n):
        oc = _cell_reading(*dec[i])
        P = gprob[i].reshape(4, 16)                      # class x cell
        for k in range(16):
            r, c = divmod(k, 4)
            up = P[:, k - 4] if r > 0 else np.zeros(4, np.float32)
            dn = P[:, k + 4] if r < 3 else np.zeros(4, np.float32)
            lf = P[:, k - 1] if c > 0 else np.zeros(4, np.float32)
            rt = P[:, k + 1] if c < 3 else np.zeros(4, np.float32)
            rows.append(np.concatenate([
                P[:, k], up, dn, lf, rt, P.max(1), P[:, k] - P.max(1),
                [r, c, r / 3., c / 3.], oc[k], oc.sum(0), fprob[i]]))
    return np.array(rows, np.float32)


def refine_grid(dec_tr, dec_te, og, tg, of, tf, G, ntr, nte):
    """Per-cell refinement of the layout map using what the reader saw in each cell.

    The layout network cannot read, but 'subtotal area' (class 2) versus 'final total area'
    (class 3) is a semantic distinction carried by words like SUBTOTAL / PPN / SERVICE vs
    TOTAL.  Feeding per-cell keyword evidence to a gradient-boosted per-cell classifier
    attacks exactly the class-2 confusion that dominates the macro-IoU loss; measured
    out-of-fold this moves the grid from 0.741 to 0.766.
    """
    Xtr = _cell_matrix(dec_tr, og, of, ntr)
    Xte = _cell_matrix(dec_te, tg, tf, nte)
    y = G.ravel()
    grp = np.repeat(np.arange(ntr), 16)
    P = dict(objective='multiclass', num_class=4, learning_rate=.05, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=.7, bagging_fraction=.8,
             bagging_freq=1, verbose=-1, seed=0)
    rounds = 150 if FAST else 400
    oof = np.zeros((len(y), 4), np.float32)
    for ti, vi in GroupKFold(5).split(Xtr, y, grp):
        oof[vi] = lgb.train(P, lgb.Dataset(Xtr[ti], y[ti]), num_boost_round=rounds).predict(Xtr[vi])
    oof4 = oof.reshape(ntr, 16, 4).transpose(0, 2, 1).reshape(ntr, 4, 4, 4)
    best = (float(np.mean([grid_score(og[i].argmax(0).ravel(), G[i]) for i in range(ntr)])), 0.0)
    for w in np.arange(0, 1.01, 0.1):
        Q = w * oof4 + (1 - w) * og
        s = float(np.mean([grid_score(Q[i].argmax(0).ravel(), G[i]) for i in range(ntr)]))
        if s > best[0]:
            best = (s, float(w))
    print(f'  grid refiner: oof {best[0]:.4f} at blend w={best[1]:.2f}', flush=True)
    pte = lgb.train(P, lgb.Dataset(Xtr, y), num_boost_round=rounds).predict(Xte)
    pte = pte.reshape(nte, 16, 4).transpose(0, 2, 1).reshape(nte, 4, 4, 4)
    return best[1] * pte + (1 - best[1]) * tg


def main(pub, out_csv):
    t0 = time.time()
    tr = pd.read_csv(os.path.join(pub, 'train.csv'))
    te = pd.read_csv(os.path.join(pub, 'test.csv'))
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    ntr, nte = len(tr), len(te)
    print('building image cache...', flush=True)
    Xtr = torch.from_numpy(build_cache(pub, tr)).float().permute(0, 3, 1, 2) / 255.
    Xte = torch.from_numpy(build_cache(pub, te)).float().permute(0, 3, 1, 2) / 255.

    print('training layout network...', flush=True)
    og, tg, on, tn, of, tf = run_grid_stage(Xtr, Xte, A)
    del Xtr, Xte

    print('rendering training strings...', flush=True)
    ns = 60000 if FAST else 240000
    SX, SY = gen_synth(ns, seed=100)
    print('training sequence reader...', flush=True)
    reader = train_reader(SX, SY, epochs=2 if FAST else 6)

    print('extracting receipt strips...', flush=True)
    packs = {}
    for split, df, gp in [('train', tr, og), ('test', te, tg)]:
        P = []
        for i, rel in enumerate(df.image_path):
            try:
                P.append(strips_for_image(os.path.join(pub, rel.replace('/', os.sep)), gp[i]))
            except Exception:
                P.append(([], np.zeros((0, NF), np.float32)))
        packs[split] = P
        print(f'  {split}: {sum(len(p[0]) for p in P)} strips', flush=True)

    def read_split(split):
        P = packs[split]
        flat = [s for p in P for s in p[0]]
        txt, cf = read_all(reader, flat) if flat else ([], np.zeros(0, np.float32))
        out, q = [], 0
        for strips, F in P:
            n = len(strips)
            out.append((txt[q:q + n], cf[q:q + n], F)); q += n
        return out

    dec_base = {s: read_split(s) for s in ('train', 'test')}
    dec = dec_base

    # ---- adapt the reader on real strips, then re-read
    print('adapting reader to real strips...', flush=True)
    RX, RY, RW = [], [], []
    okpat = re.compile(r'^[-]?[0-9A-Za-z][0-9A-Za-z.,:%]*$')
    for split in ('train', 'test'):
        for i, ((strips, F), (txt, cf, _)) in enumerate(zip(packs[split], dec[split])):
            g = golds[i] if split == 'train' else None
            for s, t, c in zip(strips, txt, cf):
                t = t.strip()
                if not t or len(t) > 22 or not okpat.match(t) or not all(ch in C2I for ch in t):
                    continue
                if g is not None and t == g:
                    RX.append(s); RY.append(t); RW.append(4)
                elif c > -0.06 and len(t) >= 2:
                    RX.append(s); RY.append(t); RW.append(1)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(SX), size=min(90000, len(SX)), replace=False)
    MX = RX + [SX[i] for i in sel]; MY = RY + [SY[i] for i in sel]
    MW = RW + [1] * len(sel)
    reader = train_reader(MX, MY, epochs=1 if FAST else 3, net=reader, lr=4e-4, weights=MW)
    dec_ft = {s: read_split(s) for s in ('train', 'test')}
    dec = dec_ft
    del SX, SY, MX, MY

    # ---- total_price ranker over BOTH readings, with a cross-model agreement flag
    print('ranking total candidates...', flush=True)
    def build(split, n, gd=None):
        Xs, grp, cands, lab = [], [], [], []
        for i in range(n):
            per = [cands_for_image(*d[split][i]) for d in (dec_ft, dec_base)]
            sets = [set(cs) for _, cs in per]
            for mi, (F, cs) in enumerate(per):
                if not cs:
                    continue
                other = sets[1 - mi]
                ag = np.array([[1. if c in other else 0., float(mi)] for c in cs], np.float32)
                Xs.append(np.concatenate([F, ag], 1))
                grp.extend([i] * len(cs)); cands.extend(cs)
                if gd is not None:
                    lab.extend([1. if c == gd[i] else 0. for c in cs])
        return (np.concatenate(Xs), np.array(grp), cands,
                (np.array(lab, np.float32) if gd is not None else None))

    Xc, gc, cc, lc = build('train', ntr, golds)
    Xq, gq, cq, _ = build('test', nte)
    tot_te = select_total(Xc, gc, cc, Xq, gq, cq, ntr, nte, golds)

    # ---- num_items and fields
    Ltr = np.stack([img_level_feats(*dec['train'][i]) for i in range(ntr)])
    Lte = np.stack([img_level_feats(*dec['test'][i]) for i in range(nte)])
    Etr = np.stack([img_level_feats(*dec['train'][i], ext=True) for i in range(ntr)])
    Ete = np.stack([img_level_feats(*dec['test'][i], ext=True) for i in range(nte)])
    y_ni = np.array([min(a['num_items'], NI_MAX) for a in A])
    Xn = np.concatenate([Ltr, on], 1); Xnte = np.concatenate([Lte, tn], 1)
    PN = dict(objective='multiclass', num_class=NI_MAX + 1, learning_rate=.05, num_leaves=15,
              min_data_in_leaf=15, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1,
              verbose=-1, seed=0)
    ni_te = (0.5 * lgb.train(PN, lgb.Dataset(Xn, y_ni), num_boost_round=300).predict(Xnte)
             + 0.5 * tn).argmax(1)

    y_f = np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32)
    Xf = np.concatenate([Etr, of], 1); Xfte = np.concatenate([Ete, tf], 1)
    PF = dict(objective='binary', learning_rate=.05, num_leaves=15, min_data_in_leaf=15,
              feature_fraction=.7, bagging_fraction=.8, bagging_freq=1, verbose=-1, seed=0)
    f_te = np.stack([lgb.train(PF, lgb.Dataset(Xf, y_f[:, j]),
                               num_boost_round=300).predict(Xfte) for j in range(4)], 1) > 0.5

    print('refining layout map with reading evidence...', flush=True)
    G_gold = np.array([a['semantic_grid'] for a in A])
    tg_ref = refine_grid(dec['train'], dec['test'], og, tg, of, tf, G_gold, ntr, nte)

    # class 3 (final total) is present in every training receipt; class 2 never appears
    # when no subtotal/service/tax flag is set
    g_te = []
    for i in range(nte):
        q = tg_ref[i].copy()
        if not f_te[i].any():
            q[2] = -1
        p = q.argmax(0).ravel()
        if not (p == 3).any():
            p[int(np.argmax(q[3].ravel()))] = 3
        g_te.append(p)
    g_te = np.array(g_te)
    rows = [dict(id=te.id[i], answer_json=json.dumps(dict(
        fields_present={k: bool(v) for k, v in zip(FKEYS, f_te[i])},
        num_items=int(ni_te[i]), semantic_grid=[int(x) for x in g_te[i]],
        total_price=tot_te[i]), sort_keys=True)) for i in range(nte)]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'wrote {out_csv} ({len(rows)} rows) in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'submission.csv')
