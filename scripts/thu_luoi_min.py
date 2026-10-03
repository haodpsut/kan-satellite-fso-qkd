#!/usr/bin/env python3
"""Thu nghiem quyet dinh: LUOI THO co phai nguyen nhan lam muc tieu thanh HANG SO?

⛔ BOI CANH. Du lieu cum cho mu = 0,95 (dung tran luoi) va beta_i = 1,6125 o MOI mau,
nen KAN khong co gi de hoc va phan ky 20/20 hat giong (R0).

Bo sinh goc dung luoi THO:
    mu_grid  = linspace(0.4, 0.95, 8)     -> buoc 0,0786, tran 0,95 (chan vat ly la 1,0)
    n_beta   = 17 tren [0,3 ; 4,5]        -> buoc 0,2625

Phep do rieng tren bai toan MOT lien ket cho thay beta* bien thien chi khoang 0,38
BUOC LUOI. Neu dieu do cung dung cho bai toan CUM thi hang so kia la HIEN VAT CUA LUOI,
khong phai tinh chat vat ly, va **duong A song**: chi can sinh lai voi luoi min.

Tep nay chay chinh solve_cluster tren CUNG cach lay mau cua bo sinh, hai cau hinh:
    THO : y het bai da cong bo
    MIN : mu toi 0,999 va nhieu diem hon, beta min hon
roi dem SO GIA TRI KHAC NHAU cua mu, chi, beta_A, beta_i o moi cau hinh.

⛔ Doi chung: cau hinh THO phai TAI LAP duoc hang so da cong bo (mu 1 gia tri,
beta_i 1 gia tri). Neu khong tai lap duoc thi phep do nay khong noi ve bai that.

    python scripts/thu_luoi_min.py --n 25
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satqkd import SystemParams, build_link_state, make_expectation   # noqa: E402
from satqkd import NodeState, solve_cluster                           # noqa: E402

CAU_HINH = {
    "THO (y bai da cong bo)": dict(mu_grid=np.linspace(0.4, 0.95, 8),
                                   chi_grid=np.array([0.0, 0.5, 1.0]),
                                   n_beta=17),
    "MIN (mu toi 0,999)":     dict(mu_grid=np.linspace(0.4, 0.999, 40),
                                   chi_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                                   n_beta=81),
}


def to_node(z, p):
    ls = build_link_state(float(np.clip(z, 0.0, 60.0)), p)
    return NodeState(ls.gamma0, ls.sigma_X, ls.w_eq)


def chay(ten, kw, n, seed, p, expect):
    rng = np.random.default_rng(seed)       # CUNG hat giong -> cung cac cum
    mus, chis, bAs, bis = [], [], [], []
    t0 = time.time()
    nf = 0
    for c in range(n):
        base = float(rng.uniform(0.0, 8.0))
        N = int(rng.integers(2, 5))
        d_eve = float(rng.uniform(22.0, 42.0))
        alice = to_node(base + rng.uniform(0.0, 2.0), p)
        bobs = [to_node(z, p) for z in base + rng.uniform(0.0, 5.0, size=N)]
        sol = solve_cluster(alice, bobs, expect, d_eve=d_eve,
                            mu_grid=kw["mu_grid"], chi_grid=kw["chi_grid"],
                            n_beta=kw["n_beta"], max_iter=8)
        if sol.n_feasible < 1:
            continue
        nf += 1
        mus.append(sol.mu); chis.append(sol.chi); bAs.append(sol.beta_A)
        bis.extend(sol.betas)

    def tom(v):
        if not v:
            return "  (khong co mau)"
        a = np.asarray(v, dtype=float)
        return ("%3d gia tri khac nhau | min %.4f max %.4f sd %.3g"
                % (len(set(np.round(a, 9))), a.min(), a.max(), a.std()))

    print("\n--- %s ---  %d/%d cum kha thi, %.0fs" % (ten, nf, n, time.time() - t0))
    print("    mu     %s" % tom(mus))
    print("    chi    %s" % tom(chis))
    print("    beta_A %s" % tom(bAs))
    print("    beta_i %s" % tom(bis))
    return dict(mu=mus, chi=chis, beta_A=bAs, beta_i=bis, nf=nf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = SystemParams()
    expect = make_expectation(p.gh_order)

    print("== Luoi THO co phai nguyen nhan lam muc tieu thanh hang so? ==")
    print("   %d cum, cung hat giong %d cho ca hai cau hinh" % (a.n, a.seed))

    ra = {}
    for ten, kw in CAU_HINH.items():
        ra[ten] = chay(ten, kw, a.n, a.seed, p, expect)

    tho = ra["THO (y bai da cong bo)"]
    minl = ra["MIN (mu toi 0,999)"]

    print("\n" + "=" * 66)
    print("DOI CHUNG: cau hinh THO co tai lap duoc hang so da cong bo khong?")
    n_mu = len(set(np.round(tho["mu"], 9)))
    n_bi = len(set(np.round(tho["beta_i"], 9)))
    tai_lap = n_mu == 1 and n_bi == 1
    print("   mu %d gia tri, beta_i %d gia tri -> %s"
          % (n_mu, n_bi, "TAI LAP DUOC (tin duoc phep do)" if tai_lap
             else "KHONG tai lap, phep do nay KHONG noi ve bai that"))
    if not tai_lap:
        return 1

    print("\nKET LUAN")
    m2 = len(set(np.round(minl["mu"], 9)))
    b2 = len(set(np.round(minl["beta_i"], 9)))
    print("   luoi MIN: mu %d gia tri, beta_i %d gia tri" % (m2, b2))
    if m2 > 1 and b2 > 1:
        print("   ✅ Luoi THO LA nguyen nhan. Duong A song: sinh lai voi luoi min")
        print("      thi muc tieu co phuong sai that va dong gop 3 dung vung.")
    elif m2 > 1 or b2 > 1:
        print("   ⚠ Chi MOT trong hai tang co phuong sai tro lai. Duong A cuu duoc")
        print("      mot phan; tang con lai van suy bien va phai khai.")
    else:
        print("   ⛔ Luoi min KHONG lam muc tieu bien thien. Suy bien la tinh chat")
        print("      cua BAI TOAN, khong phai cua luoi => bang chung cho duong B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
