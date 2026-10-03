#!/usr/bin/env python3
"""R4-3 va R4-4: do do nhay voi khoi tao, va nhip tai toi uu khi trien khai.

R4-3 (phan bien: "convergence analysis being EMPIRICAL IS WEAK", doi monotonicity,
dieu kien ton tai diem bat dong, hoi tu dia phuong, va DO NHAY VOI KHOI TAO):
  do duoc ngay la **do nhay voi khoi tao**. Chay vong lap mean-field tu nhieu diem
  xuat phat khac nhau tren cung mot cum, xem no co ve cung mot diem bat dong khong,
  va mat bao nhieu buoc.

R4-4 (phan bien: "add a discussion on PRACTICAL DEPLOYMENT TIMING REQUIREMENTS,
HOW OFTEN re-optimization must occur, and IF OR WHEN RE-TRAINING CAN BE AVOIDED"):
  cau hoi that khong phai "giai mat bao lau" ma "tham so toi uu DOI NHANH DEN MUC NAO".
  Do: doc mot lan bay, tinh tham so toi uu o tung buoc, roi hoi giu tham so cua buoc
  truoc bao lau thi mat bao nhieu phan tram toc do khoa.

⛔ Moi con so phai sinh tren sol1 (may cua ho so), khong phai tren Mac.

    python3 scripts/do_trien_khai.py
"""
import sys
import time
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


def phan_1_khoi_tao(p, expect, n_cum=12, seed=0):
    """R4-3: cung mot cum, nhieu diem khoi tao khac nhau cho vong lap trong."""
    rng = np.random.default_rng(seed)
    print("== R4-3: do nhay cua diem bat dong voi KHOI TAO ==")
    print("   %d cum, moi cum chay lai voi 5 beta khoi tao khac nhau\n" % n_cum)
    # ⛔ Lan dau toi truyen `beta_init=` cho solve_cluster, nhung no KHONG co tham so
    # ay, nen ca 5 "diem khoi tao" goi y het nhau va phep thu se LUON bao "khong nhay
    # voi khoi tao" du su that the nao. Phep thu vo hieu, da bo.
    #
    # Cach dung: `inner_meanfield` von da da-khoi-tao san voi init_starts=(0.6,1.2,1.8,2.4)
    # va giu cai tot nhat. Cau hoi khoa hoc that la: MOT khoi tao don le co dan toi cung
    # diem bat dong khong. Nen goi inner_meanfield voi init_starts=(x,) cho tung x.
    from satqkd.mu_solver import inner_meanfield
    khoi_tao = [0.6, 1.2, 1.8, 2.4, 3.6]
    lech_b, khac_nhau, tong_ca = [], 0, 0
    for c in range(n_cum):
        base = float(rng.uniform(0.0, 8.0))
        N = int(rng.integers(2, 5))
        d_eve = float(rng.uniform(22.0, 42.0))
        alice = to_node(base + rng.uniform(0.0, 2.0), p)
        bobs = [to_node(z, p) for z in base + rng.uniform(0.0, 5.0, size=N)]
        # chot (mu, chi) o diem toi uu de chi do rieng vong lap TRONG
        sol = solve_cluster(alice, bobs, expect, d_eve=d_eve, mu_grid=MU_G,
                            chi_grid=CHI_G, n_beta=17, max_iter=8)
        if sol.n_feasible < 1:
            continue
        ket = []
        for b0 in khoi_tao:
            bA, bs, res = inner_meanfield(sol.mu, sol.chi, alice, bobs, expect,
                                          d_eve=d_eve, n_beta=17, max_iter=8,
                                          init_starts=(b0,))
            ket.append((bA, tuple(np.round(bs, 12)), res["n_feasible"]))
        tong_ca += 1
        print("     cum %d/%d xong" % (tong_ca, n_cum), flush=True)
        lech = max(k[0] for k in ket) - min(k[0] for k in ket)
        lech_b.append(lech)
        if len({k[1] for k in ket}) > 1 or lech > 1e-9:
            khac_nhau += 1
    if not lech_b:
        print("   (khong co cum kha thi)")
        return
    v = np.array(lech_b)
    print("   bien thien beta_A* giua cac khoi tao DON LE: max %.4g, trung vi %.4g"
          % (v.max(), float(np.median(v))))
    print("   so cum cho diem bat dong KHAC NHAU tuy khoi tao: %d/%d" % (khac_nhau, tong_ca))
    if khac_nhau == 0:
        print("\n   ⇒ Tren cac cum da thu, mot khoi tao don le bat ky deu ve CUNG diem")
        print("     bat dong. Day la bang chung THUC NGHIEM, khong phai chung minh.")
    else:
        print("\n   ⛔ CO %d/%d cum cho diem bat dong khac nhau tuy khoi tao." % (khac_nhau, tong_ca))
        print("     Dieu nay GIAI THICH vi sao ma phai da-khoi-tao, va tuyen bo 'duy nhat'")
        print("     trong bai phai duoc rut lai hoac gioi han dieu kien.")


def phan_2_nhip(p, expect, n_buoc=40, dt=50.0, n_quy_dao=8):
    """R4-4: giu tham so cua buoc truoc bao lau thi mat bao nhieu toc do khoa.

    ⛔ Ban dau toi chay MOT quy dao, MOT hat giong, va duong cong ra KHONG DON DIEU
    (giu 8 buoc cho 84,8% trong khi giu 5 buoc chi 74%). Do la nhieu at xu the, khong
    phai mot phat hien. Nay trung binh tren %d quy dao va in ca do lech chuan, de biet
    con so nao doc duoc.
    """
    print("\n== R4-4: nhip TAI TOI UU khi trien khai ==")
    print("   %d quy dao x %d buoc, moi buoc %.0f s\n" % (n_quy_dao, n_buoc, dt))
    giu_ds = (1, 2, 3, 5, 8, 12, 20, n_buoc)
    thu = {g: [] for g in giu_ds}
    t_oracle = []
    for q in range(n_quy_dao):
        rng = np.random.default_rng(100 + q)
        dinh = float(rng.uniform(35.0, 58.0))
        z = np.abs(np.linspace(-1, 1, n_buoc)) * dinh + float(rng.uniform(2.0, 8.0))
        N = int(rng.integers(2, 5))
        d_eve = float(rng.uniform(22.0, 42.0))
        trang_thai, toi_uu = [], []
        t0 = time.time()
        for k in range(n_buoc):
            alice = to_node(z[k], p)
            bobs = [to_node(z[k] + float(rng.uniform(0, 4)), p) for _ in range(N)]
            sol = solve_cluster(alice, bobs, expect, d_eve=d_eve, mu_grid=MU_G,
                                chi_grid=CHI_G, n_beta=17, max_iter=8)
            trang_thai.append((alice, bobs)); toi_uu.append(sol)
        t_oracle.append((time.time() - t0) / n_buoc)

        def rate(sol, st):
            alice, bobs = st
            r = cluster_key_rate(sol.mu, sol.beta_A, sol.betas, sol.chi, alice, bobs,
                                 expect, d_eve=d_eve)
            return sum(x for x, f in zip(r["per_rate"], r["per_feasible"]) if f)

        moc = sum(rate(toi_uu[k], trang_thai[k]) for k in range(n_buoc))
        if moc <= 0:
            print("     quy dao %d/%d: khong co khoa, bo qua" % (q + 1, n_quy_dao), flush=True)
            continue
        for g in giu_ds:
            tong = sum(rate(toi_uu[(k // g) * g], trang_thai[k]) for k in range(n_buoc))
            thu[g].append(100.0 * tong / moc)
        print("     quy dao %d/%d xong" % (q + 1, n_quy_dao), flush=True)

    print("\n   chi phi oracle: %.2f +- %.2f s / buoc"
          % (float(np.mean(t_oracle)), float(np.std(t_oracle))))
    print("\n   %-22s %18s" % ("tai toi uu moi", "giu lai (%)"))
    print("   " + "-" * 42)
    for g in giu_ds:
        v = np.array(thu[g])
        if not len(v):
            continue
        print("   moi %2d buoc (%4.0f s)   %7.1f +- %5.1f   (n=%d)"
              % (g, g * dt, v.mean(), v.std(), len(v)))
    print("\n   Doc: nhip tai toi uu du dung la dong CUOI CUNG con gan 100%%.")
    print("   So sanh voi %.1f s/buoc cua oracle de biet bo dieu khien hoc co can"
          % float(np.mean(t_oracle)))
    print("   thiet vi TOC DO hay khong; neu khong thi ly do phai la cho khac.")


def main():
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    phan_1_khoi_tao(p, expect)
    phan_2_nhip(p, expect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
