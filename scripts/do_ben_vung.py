#!/usr/bin/env python3
"""R1-3: do BEN VUNG tren bon truc ma phan bien neu ra.

Phan bien 1 diem 3: "additional studies under different atmospheric conditions,
satellite geometries, user distributions, or orbital configurations".

Bon truc, moi truc quet quanh diem van hanh cua bai, va o moi diem hoi hai cau:
  1. toi uu co con nam o cung cho khong (mu* van cham tran? beta* dich bao nhieu?)
  2. loi ich cua thich ung so voi tham so TINH tot nhat con bao nhieu

Cau 2 moi la cai dang bao: no do xem KET LUAN cua bai co song khi doi dieu kien khong,
chu khong chi do rang so co doi hay khong.

⛔ Chay tren sol1.

    python3 scripts/do_ben_vung.py
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satqkd import (SystemParams, build_link_state, make_expectation,   # noqa: E402
                    NodeState, solve_cluster, cluster_key_rate)

MU_G = np.linspace(0.4, 0.95, 12)
CHI_G = np.linspace(0.0, 1.0, 5)


def to_node(z, p):
    ls = build_link_state(float(np.clip(z, 0.0, 60.0)), p)
    return NodeState(ls.gamma0, ls.sigma_X, ls.w_eq)


def mot_dieu_kien(p, expect, n_cum, rng, z_lo, z_hi, n_user, d_lo, d_hi):
    """Tra ve (ty le mu* cham tran, bien thien beta*, loi ich thich ung so voi tinh)."""
    cham_tran, betas, loi_ich = 0, [], []
    n_ok = 0
    for _ in range(n_cum):
        base = float(rng.uniform(z_lo, z_hi))
        N = n_user if isinstance(n_user, int) else int(rng.integers(*n_user))
        d_eve = float(rng.uniform(d_lo, d_hi))
        alice = to_node(base + rng.uniform(0.0, 2.0), p)
        bobs = [to_node(base + rng.uniform(0.0, 5.0), p) for _ in range(N)]
        sol = solve_cluster(alice, bobs, expect, d_eve=d_eve, mu_grid=MU_G,
                            chi_grid=CHI_G, n_beta=17, max_iter=8)
        if sol.n_feasible < 1:
            continue
        n_ok += 1
        if sol.mu >= MU_G[-1] - 1e-12:
            cham_tran += 1
        betas.append(sol.beta_A)

        def rate(s):
            r = cluster_key_rate(s.mu, s.beta_A, s.betas, s.chi, alice, bobs,
                                 expect, d_eve=d_eve)
            return sum(x for x, f in zip(r["per_rate"], r["per_feasible"]) if f)
        thich_ung = rate(sol)
        # tham so TINH tot nhat: diem giua luoi, co dinh cho moi cum
        class S:
            mu, chi = 0.95, 1.0
            beta_A = 1.6125
            betas = [1.6125] * N
        tinh = rate(S())
        if tinh > 0:
            loi_ich.append(thich_ung / tinh)
        elif thich_ung > 0:
            loi_ich.append(float("inf"))
    if not n_ok:
        return None
    return (100.0 * cham_tran / n_ok,
            (max(betas) - min(betas)) if betas else 0.0,
            float(np.median([x for x in loi_ich if np.isfinite(x)])) if loi_ich else float("nan"),
            n_ok)


def main():
    expect0 = make_expectation(SystemParams().gh_order)
    n_cum = 25
    print("== R1-3: ben vung tren bon truc ==")
    print("   %d cum moi dieu kien, tham so oracle\n" % n_cum)
    print("   %-34s %9s %11s %11s %6s"
          % ("dieu kien", "mu* tran", "bien beta*", "loi ich", "n"))
    print("   " + "-" * 76)

    ca = []
    # 1. khi quyen: gio (anh huong turbulence qua Hufnagel-Valley)
    for w in (11.0, 21.0, 31.0, 41.0):
        ca.append(("khi quyen: gio %.0f m/s" % w,
                   replace(SystemParams(), wind_rms=w), dict(z_lo=0, z_hi=8, n_user=(2, 5),
                                                             d_lo=22, d_hi=42)))
    # 2. hinh hoc: dai zenith
    for lo, hi in ((0, 4), (0, 8), (8, 20), (20, 35)):
        ca.append(("hinh hoc: zenith %d-%d do" % (lo, hi), SystemParams(),
                   dict(z_lo=lo, z_hi=hi, n_user=(2, 5), d_lo=22, d_hi=42)))
    # 3. phan bo nguoi dung: so nguoi
    for n in (2, 3, 4, 6):
        ca.append(("nguoi dung: N = %d" % n, SystemParams(),
                   dict(z_lo=0, z_hi=8, n_user=n, d_lo=22, d_hi=42)))
    # 4. cau hinh Eve / quy dao: khoang cach Eve
    for lo, hi in ((10, 20), (22, 42), (42, 70), (70, 120)):
        ca.append(("Eve: d = %d-%d m" % (lo, hi), SystemParams(),
                   dict(z_lo=0, z_hi=8, n_user=(2, 5), d_lo=lo, d_hi=hi)))

    t0 = time.time()
    for i, (ten, p, kw) in enumerate(ca, 1):
        ex = make_expectation(p.gh_order)
        rng = np.random.default_rng(7)      # CUNG hat giong moi dieu kien
        r = mot_dieu_kien(p, ex, n_cum, rng, **kw)
        if r is None:
            print("   %-34s %9s %11s %11s %6d" % (ten, "-", "-", "khong kha thi", 0))
        else:
            tran, bien, loi, n = r
            print("   %-34s %8.0f%% %11.4f %10.2fx %6d" % (ten, tran, bien, loi, n))
        print("     (%d/%d, %.0f s)" % (i, len(ca), time.time() - t0), flush=True)
    print("\n   Doc: cot 'mu* tran' cho biet su bao hoa co phai dac tinh cua CHE DO hay")
    print("   cua CA BAI TOAN. Cot 'loi ich' cho biet ket luan trung tam cua bai (thich")
    print("   ung hon tinh) co song qua cac dieu kien hay khong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
