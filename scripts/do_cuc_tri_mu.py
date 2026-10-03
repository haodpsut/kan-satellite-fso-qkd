#!/usr/bin/env python3
"""PHEP DO QUYET DINH giua duong A va duong B cua goi R0.

⛔ CAU HOI. Bang IV cua bai noi bo dieu khien KAN du doan (mu*, chi*, beta*). Nhung
du lieu cum cho **mu* = 0,95 o MOI mau**, dung bang tran luoi, va beta_i cung MOT gia
tri. R0 da do: KAN phan ky 20/20 hat giong vi muc tieu la hang so.

Hai duong:
  A  sinh lai du lieu o che do CO PHUONG SAI  -> cuu nguyen dong gop 3
  B  doi trong tam: toi uu BAO HOA o tran dieu che -> bac B4, phai Hao duyet

Chon duong nao KHONG duoc doan. Tep nay hoi dung mot cau tra loi duoc bang so:

  **Co ton tai che do vat ly nao ma mu* nam HAN BEN TRONG (0,1) khong?**

docs/formulation.md: mu la DO SAU DIEU CHE, chan vat ly trong (0,1), va gamma ~ mu.
Neu khong rang buoc nao can thi muc tieu DON DIEU tang theo mu, cuc tri luon o bien
va duong A chet vi VAT LY chu khong vi tinh toan. Nhung gamma cua Eve cung phu thuoc
mu, nen rang buoc eve_error > 0.1 CO THE can va tao cuc tri trong. Phai do.

Cach do, khong dung luoi tho cua bo sinh:
  1. quet mu tren luoi MIN tu 0,05 toi 0,999 (501 diem), cho nhieu dieu kien kenh
  2. voi moi dieu kien, tim mu* kha thi va hoi: no o BIEN hay BEN TRONG
  3. ghi ca ty le rang buoc nao DANG can tai mu*

Doi chung: cung ham do chay tren mot bai toan CO cuc tri trong da biet (ham loi
nhan tao) phai tra ve "BEN TRONG"; neu khong thi thuoc hong, khong phai ket qua.

    python scripts/do_cuc_tri_mu.py --n 200
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from satqkd import SystemParams, build_link_state, make_expectation   # noqa: E402
from satqkd.detection import sift_qber, eve_error                     # noqa: E402
from satqkd.keyrate import normalized_key_rate                        # noqa: E402

MU_LO, MU_HI, MU_N = 0.05, 0.999, 501
QBER_MAX, PSIFT_MIN, EVE_MIN = 1e-3, 1e-3, 0.1


def quet_mu(ls, expect, d_eve, beta, mu_vals):
    """Tra ve (rate, feasible, ly_do_khong_kha_thi) cho tung mu."""
    ra = []
    for mu in mu_vals:
        psift, qber = sift_qber(ls.gamma(mu), beta, ls.sigma_X, expect)
        pe = eve_error(ls.gamma_eve(mu, d_eve), ls.sigma_X, expect)
        r = normalized_key_rate(psift, qber, pe)
        vi_pham = []
        if not (qber < QBER_MAX):
            vi_pham.append("qber")
        if not (psift > PSIFT_MIN):
            vi_pham.append("psift")
        if not (pe > EVE_MIN):
            vi_pham.append("eve")
        ra.append((float(r), not vi_pham, vi_pham))
    return ra


def phan_loai(mu_vals, quet, bien_eps=2):
    """mu* nam o BIEN hay BEN TRONG? bien_eps = so o luoi tinh la 'sat bien'."""
    kha = [i for i, (_, f, _) in enumerate(quet) if f]
    if not kha:
        return "khong kha thi", None, None
    i = max(kha, key=lambda j: quet[j][0])
    if i <= bien_eps:
        return "BIEN duoi", mu_vals[i], quet[i]
    if i >= len(mu_vals) - 1 - bien_eps:
        return "BIEN tren", mu_vals[i], quet[i]
    return "BEN TRONG", mu_vals[i], quet[i]


def doi_chung():
    """Thuoc phai tra ve BEN TRONG tren mot bai toan CO cuc tri trong da biet."""
    mu = np.linspace(MU_LO, MU_HI, MU_N)
    gia = [(-(m - 0.5) ** 2, True, []) for m in mu]      # dinh o 0,5
    loai, mstar, _ = phan_loai(mu, gia)
    ok1 = loai == "BEN TRONG" and abs(mstar - 0.5) < 0.01
    gia2 = [(m, True, []) for m in mu]                   # don dieu tang
    loai2, _, _ = phan_loai(mu, gia2)
    ok2 = loai2 == "BIEN tren"
    print("== DOI CHUNG THUOC ==")
    print("  ham co dinh o 0,5        -> %-12s %s" % (loai, "DAT" if ok1 else "HONG"))
    print("  ham don dieu tang        -> %-12s %s" % (loai2, "DAT" if ok2 else "HONG"))
    return ok1 and ok2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="so dieu kien kenh lay mau")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rong", action="store_true",
                    help="lay mau RONG hon bo sinh goc (zenith va d_eve rong hon)")
    a = ap.parse_args()

    if not doi_chung():
        print("\n⛔ THUOC HONG, dung doc ket qua ben duoi.")
        return 1
    print()

    p = SystemParams()
    expect = make_expectation(p.gh_order)
    rng = np.random.default_rng(a.seed)
    mu_vals = np.linspace(MU_LO, MU_HI, MU_N)

    # Luoi beta giong bo sinh goc (n_beta=17 tren dai cua no)
    beta_vals = np.linspace(0.3, 4.5, 17)

    dem = {}
    can = {}
    mstars = []
    for k in range(a.n):
        if a.rong:
            z = float(rng.uniform(0.0, 40.0))
            d_eve = float(rng.uniform(5.0, 80.0))
        else:
            z = float(rng.uniform(0.0, 13.0))        # nhu bo sinh: base 0-8 + 0-5
            d_eve = float(rng.uniform(22.0, 42.0))
        ls = build_link_state(z, p)

        tot = None
        for b in beta_vals:
            q = quet_mu(ls, expect, d_eve, float(b), mu_vals)
            loai, mstar, cell = phan_loai(mu_vals, q)
            if loai == "khong kha thi":
                continue
            if tot is None or cell[0] > tot[2][0]:
                tot = (loai, mstar, cell)
        if tot is None:
            dem["khong kha thi"] = dem.get("khong kha thi", 0) + 1
            continue
        loai, mstar, cell = tot
        dem[loai] = dem.get(loai, 0) + 1
        mstars.append(mstar)
        # rang buoc nao dang can NGAY TREN mu* (thu nhich len mot o)
        i = int(np.argmin(np.abs(mu_vals - mstar)))
        if i + 1 < len(mu_vals):
            q = quet_mu(ls, expect, d_eve, float(beta_vals[0]), mu_vals[i + 1:i + 2])
            for v in q[0][2]:
                can[v] = can.get(v, 0) + 1

    print("== KET QUA: mu* nam o dau, tren %d dieu kien kenh ==" % a.n)
    print("   (luoi mu: %d diem tu %.3f toi %.3f, chan vat ly la 1,0)\n"
          % (MU_N, MU_LO, MU_HI))
    tong = sum(dem.values())
    for k in sorted(dem, key=lambda x: -dem[x]):
        print("   %-16s %4d  (%5.1f%%)" % (k, dem[k], 100.0 * dem[k] / tong))
    if mstars:
        print("\n   mu* : min %.4f  max %.4f  trung vi %.4f"
              % (min(mstars), max(mstars), float(np.median(mstars))))
    if can:
        print("\n   rang buoc dang can ngay tren mu*: %s" % can)

    ben_trong = dem.get("BEN TRONG", 0)
    print("\n" + "=" * 62)
    if ben_trong == 0:
        print("  ⛔ KHONG co dieu kien nao cho cuc tri TRONG.")
        print("     mu* luon cham bien. Duong A (sinh lai de co phuong sai theo mu)")
        print("     KHONG cuu duoc, vi muc tieu don dieu theo mu tren CA mien vat ly.")
        print("     => bang chung cho duong B.")
    else:
        print("  ✅ CO %d/%d dieu kien cho cuc tri TRONG (%.1f%%)."
              % (ben_trong, tong, 100.0 * ben_trong / tong))
        print("     Duong A kha thi: sinh lai du lieu TAP TRUNG vao che do nay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
