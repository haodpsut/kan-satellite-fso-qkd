#!/usr/bin/env python3
"""Thay cong KHA THI nhi phan bang BIEN DO, va in kem DAI ON DINH cua con so.

⛔ VI SAO. Do duoc 02/10: dem so cap kha thi trong vong kin **lat o chu so thu 15**.
Cung mot mo hinh tat dinh, cung du lieu, cung ma:

    nhieu tuong doi 0      -> 91/158
    nhieu tuong doi 1e-16  -> 91/158
    nhieu tuong doi 1e-15  -> 97/158      (dung bang so cua sol1)
    nhieu tuong doi 1e-12  -> 158/158     (dung bang so cua ban da sua)

Nguyen nhan: oracle dat beta **dung tren mat rang buoc BSA** (`bsa_i >= 0.005`), nen
mot cap kha thi hay khong do chu so thap phan thu 15 quyet dinh. Day cung la thu
Reviewer 2 da thay: *"do lech chuan KeyRet cung bac voi gia tri"*.

⇒ Mot con so dem nhi phan tai day KHONG phai phep do. Tep nay bao cao thay bang:
  1. BIEN DO toi tung rang buoc, phan phoi, va rang buoc nao dang can
  2. so cap kha thi nhu MOT HAM cua dung sai, de thay con so on dinh tu dau
  3. DAI ON DINH: dem o nhieu +-1e-12, in kem con so chinh

⛔ Khong sua `cluster_key_rate`: do la hien vat cua bai. Tep nay do THEM, khong doi
vat ly.

    python scripts/do_bien_do_kha_thi.py --models linear,mlp,kan --seeds 5
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_cluster as tc                                          # noqa: E402
from satqkd import SystemParams, make_expectation, NodeState        # noqa: E402
from satqkd.multiuser import (link_stats, node_eve_error,           # noqa: E402
                              bsa_deviation, exclusion_term)
from satqkd.keyrate import binary_entropy as H2                     # noqa: E402

TEN_RB = ["sigma>0", "P_chi>psift_min", "leak<=leak_max",
          "peE_A>eve_min", "peE_i>eve_min", "bsa_A>=bsa_min", "bsa_i>=bsa_min"]


def chia_tap(g_rows, u_rows):
    """Y HET train_cluster.main: 20% cum DAU sau khi xao lam test."""
    cids = sorted({int(r["cluster_id"]) for r in g_rows})
    rng = np.random.default_rng(0)
    rng.shuffle(cids)
    nte = max(1, int(0.2 * len(cids)))
    te, tr = set(cids[:nte]), set(cids[nte:])
    return ([r for r in g_rows if int(r["cluster_id"]) in tr],
            [r for r in g_rows if int(r["cluster_id"]) in te],
            [r for r in u_rows if int(r["cluster_id"]) in tr])


def bien_do(mu, bA, bi, chi, alice, bobs, expect, d_eve,
            psift_min=1e-3, eve_min=0.1, leak_max=0.05,
            bsa_split=0.015, bsa_min=0.005):
    """Bien do (duong = thoa) cua 7 rang buoc, cho tung cap trong cum."""
    pcA, peA, psA = link_stats(alice, mu, bA, expect)
    peEA = node_eve_error(alice, mu, d_eve, expect)
    bsaA = bsa_deviation(alice, mu, bA, expect, bsa_split)
    pc, pe, ps, peE = [], [], [], []
    for nd, b in zip(bobs, bi):
        c, e, s = link_stats(nd, mu, float(b), expect)
        pc.append(c); pe.append(e); ps.append(s)
        peE.append(node_eve_error(nd, mu, d_eve, expect))
    ra = []
    for j in range(len(bobs)):
        P_AB = psA * ps[j]
        qber = (pcA * pe[j] + peA * pc[j]) / P_AB if P_AB > 0 else 1.0
        others = [pc[k] for k in range(len(bobs)) if k != j]
        P_ex = exclusion_term(pcA, pc[j], others)
        P_chi = max(P_AB - chi * P_ex, 0.0)
        leak = (1.0 - chi) * P_ex / P_chi if P_chi > 0 else float("inf")
        sig = max(0.0, (1 - H2(qber)) - max(1 - H2(peEA), 1 - H2(peE[j])))
        bsai = bsa_deviation(bobs[j], mu, float(bi[j]), expect, bsa_split)
        ra.append([sig, P_chi - psift_min, leak_max - leak,
                   peEA - eve_min, peE[j] - eve_min,
                   bsaA - bsa_min, bsai - bsa_min])
    return np.array(ra)


def chay_mot_mo_hinh(key, sd, args, g_tr, g_te, u_tr, by, expect, p):
    GX = lambda rs: np.array([[r[k] for k in tc.G_FEATS] for r in rs])   # noqa: E731
    UXf = lambda rs: np.array([[r[k] for k in tc.U_FEATS] for r in rs])  # noqa: E731
    Gn, gs = tc.standardize(GX(g_tr))
    Un, us = tc.standardize(UXf(u_tr))
    gm = tc.make(key, sd, args.device, 1500, residual=args.residual,
                 bo_suy_bien=args.bo_suy_bien)
    um = tc.make(key, sd, args.device, 1500, residual=args.residual,
                 bo_suy_bien=args.bo_suy_bien)
    gm.fit(Gn, np.array([[r[k] for k in tc.G_LABELS] for r in g_tr]), seed=sd)
    um.fit(Un, np.array([[r[k] for k in tc.U_LABELS] for r in u_tr]), seed=sd)
    pred = tc.clip_global(gm.predict(tc.standardize(GX(g_te), gs)))

    M = []
    for i, gr in enumerate(g_te):
        mu, chi, bA = [float(x) for x in pred[i]]
        urs = by[int(gr["cluster_id"])]
        alice = NodeState(gr["gA"], gr["sA"], gr["wA"])
        bobs = [NodeState(x["g_i"], x["s_i"], x["w_i"]) for x in urs]
        UXr = np.array([[x["g_i"], x["s_i"], x["w_i"], gr["gA"], gr["sA"], mu, chi]
                        for x in urs])
        bi = np.clip(um.predict(tc.standardize(UXr, us))[:, 0],
                     tc.BETA_LO, tc.BETA_HI)
        M.append(bien_do(mu, bA, bi, chi, alice, bobs, expect,
                         float(urs[0]["d_eve"])))
    return np.vstack(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="linear,mlp,kan")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--residual", action="store_true", default=True)
    ap.add_argument("--bo-muc-tieu-suy-bien", dest="bo_suy_bien",
                    action="store_true")
    a = ap.parse_args()

    g = tc._read_csv(tc.RESULTS / "cluster_global.csv")
    u = tc._read_csv(tc.RESULTS / "cluster_user.csv")
    g_tr, g_te, u_tr = chia_tap(g, u)
    by = collections.defaultdict(list)
    for r in u:
        by[int(r["cluster_id"])].append(r)
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    n_or = sum(int(r["nfeas_opt"]) for r in g_te)

    print("== BIEN DO kha thi thay cho DEM nhi phan ==")
    print("   %d cum test, oracle %d cap kha thi, %d hat giong\n"
          % (len(g_te), n_or, a.seeds))

    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        Ms = [chay_mot_mo_hinh(key, sd, a, g_tr, g_te, u_tr, by, expect, p)
              for sd in range(a.seeds)]
        M = np.stack(Ms)                       # (seed, cap, rang buoc)
        print("--- %s ---" % key)
        # dem o dung sai 0, va DAI ON DINH quanh no
        for tol in (-1e-12, 0.0, 1e-12):
            n = (M > tol).all(axis=2).sum(axis=1)
            nhan = {-1e-12: "chat hon 1e-12", 0.0: "dung sai 0",
                    1e-12: "long hon 1e-12"}[tol]
            print("    kha thi, %-16s %6.1f +- %4.1f / %d"
                  % (nhan, n.mean(), n.std(), n_or))
        # rang buoc nao dang can
        viphạm = (M <= 0).sum(axis=(0, 1))
        thu = sorted(zip(TEN_RB, viphạm), key=lambda x: -x[1])
        print("    rang buoc bi vi pham nhieu nhat: %s"
              % ", ".join("%s=%d" % (t, c) for t, c in thu[:3] if c))
        # bien do cua rang buoc can nhat
        j = int(np.argmax(viphạm))
        b = M[:, :, j].ravel()
        print("    bien do cua '%s': trung vi %+.4g | %d/%d cap co |bien do| < 1e-9"
              % (TEN_RB[j], float(np.median(b)),
                 int((np.abs(b) < 1e-9).sum()), b.size))
        print()

    print("⛔ Doc bang nay the nao: neu ba dong 'kha thi' CHENH NHAU thi con so dem")
    print("   khong phai phep do, phai bao cao bien do. Neu ba dong BANG NHAU thi")
    print("   con so dem on dinh va dung duoc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
