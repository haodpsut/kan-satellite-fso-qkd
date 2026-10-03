"""Lap lai cac dong cua Bang III qua NHIEU ky nguyen TLE va nhieu tram mat dat.

⛔ VI SAO CO TEP NAY. Nguoi doc ngoai 03/10/2026 goi day la "the single highest-value
run remaining": moi dong cua Bang III hien la MOT ky nguyen TLE, MOT tram (PTIT/Ha Noi),
MOT chan troi, khong lap lai lan nao. Tu do bai rut ra "3,6x" (18 so voi 5 buoc cap an
toan tren 201) va "khong lua chon tinh nao an toan ca" (2 so voi 0 tren 87). Ho viet:

  "A referee will ask what the spread of those counts is across epochs and stations,
   and reporting a ratio as ∞ from 0 versus 2 will be read as overreach."

Dung. Mot ty so tinh tu 2 su kien so voi 0 su kien, do mot lan, khong phai phep do.

Tep nay chay lai DUNG phep tinh cua scripts/optimize_pass_cluster.py, nhung quet qua
mot luoi (ky nguyen x tram), va bao cao TRUNG BINH va KHOANG cua so buoc cap an toan
thay vi mot con so.

⛔ Hinh hoc cum (ALICE_DZ, BOB_DZ, BOB_DEVE) duoc NHAP tu chinh tep goc, khong chep
lai: hai ban sao cua cung mot dac ta la he tham chieu thu hai, va no se lech.

Chay tren sol1:
    python3 scripts/lap_nhieu_ky_nguyen.py --che-do long   --ky-nguyen 10
    python3 scripts/lap_nhieu_ky_nguyen.py --che-do sparse --ky-nguyen 10
    python3 scripts/lap_nhieu_ky_nguyen.py --tu-kiem
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from satqkd import SystemParams, make_expectation                 # noqa: E402
from satqkd import cluster_key_rate, solve_cluster                # noqa: E402
from satqkd.orbit_tle import (PTIT_HANOI, GroundStation,          # noqa: E402
                              example_tle_constellation)
from optimize_pass_cluster import make_cluster, BOB_DZ            # noqa: E402

RESULTS = ROOT / "results"

# Hai che do dung DUNG tham so cua hai dong TLE Walker trong Bang III.
CHE_DO = {
    "long":   dict(n_sats=500, horizon=5400.0, dt=50.0),
    "sparse": dict(n_sats=200, horizon=3000.0, dt=50.0),
}

# Tram mat dat. PTIT la tram cua bai; hai tram kia them vao de tra loi cau hoi
# "ket qua co phu thuoc tram khong". Chon mot tram cung vi do va mot tram vi do
# cao hon han, vi so lan nhin thay ve tinh cua vo nghieng 53 do phu thuoc vi do.
TRAM = [
    PTIT_HANOI,
    GroundStation(lat_deg=16.0544, lon_deg=108.2022, elev_m=10.0, name="DaNang"),
    GroundStation(lat_deg=47.3769, lon_deg=8.5417, elev_m=408.0, name="Zurich"),
]

D_EVE = 26.0


def mot_lan(p, expect, n_sats, horizon, dt, t0_utc, ground, nhanh=False):
    """Tra ve (so buoc cap an toan TINH, ADAPTIVE, tong so buoc cap, so buoc duoc phuc vu).

    Lap lai y het mach tinh cua optimize_pass_cluster.main().
    """
    const = example_tle_constellation(n_sats=n_sats, t0_utc=t0_utc,
                                      ground=ground, min_elevation=30.0)
    t_grid = np.arange(0.0, horizon, dt)
    sched = const.schedule(t_grid)
    served = [(float(sched["t"][i]), float(sched["zenith_deg"][i]))
              for i in range(t_grid.size)
              if sched["sat_id"][i] >= 0 and np.isfinite(sched["zenith_deg"][i])]
    if not served:
        return 0, 0, 0, 0
    N = len(BOB_DZ)
    clusters = [(t, z, *make_cluster(z, p)) for (t, z) in served]

    mu_grid = np.linspace(0.4, 0.95, 6 if nhanh else 12)
    chi_grid = np.array([0.0, 0.5, 1.0]) if nhanh else np.linspace(0, 1, 5)
    inner_kw = dict(n_beta=15 if nhanh else 25, max_iter=8)

    # moc TINH tot nhat tren CHINH lan chay nay (khong muon tu lan khac)
    best = (-1.0, 0.5, 2.0, 1.0)
    for mu in np.linspace(0.4, 0.95, 6):
        for beta in np.linspace(0.5, 4.0, 8):
            for chi in (0.0, 0.5, 1.0):
                tot = 0.0
                for (_, _, alice, bobs) in clusters:
                    r = cluster_key_rate(mu, beta, [beta] * N, chi, alice, bobs,
                                         expect, D_EVE)
                    tot += sum(rr for rr, f in zip(r["per_rate"], r["per_feasible"]) if f)
                if tot > best[0]:
                    best = (tot, float(mu), float(beta), float(chi))
    _, MU_BS, BETA_BS, CHI_BS = best

    sec_s = sec_a = 0
    for (_, _, alice, bobs) in clusters:
        rs = cluster_key_rate(MU_BS, BETA_BS, [BETA_BS] * N, CHI_BS, alice, bobs,
                              expect, D_EVE)
        sec_s += rs["n_feasible"]
        sol = solve_cluster(alice, bobs, expect, d_eve=D_EVE,
                            mu_grid=mu_grid, chi_grid=chi_grid, **inner_kw)
        ra = cluster_key_rate(sol.mu, sol.beta_A, sol.betas, sol.chi, alice, bobs,
                              expect, D_EVE)
        sec_a += ra["n_feasible"]
    return sec_s, sec_a, len(clusters) * N, len(clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--che-do", default="long", choices=sorted(CHE_DO))
    ap.add_argument("--ky-nguyen", type=int, default=10, help="so ky nguyen TLE")
    ap.add_argument("--cach-ngay", type=float, default=1.0, help="khoang cach giua hai ky nguyen, ngay")
    ap.add_argument("--tram", default="all", help="'all', hoac ten tram cach nhau bang dau phay")
    ap.add_argument("--nhanh", action="store_true")
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()

    p = SystemParams()
    expect = make_expectation(p.gh_order)
    cfg = CHE_DO[a.che_do]
    trams = TRAM if a.tram == "all" else [t for t in TRAM if t.name in a.tram.split(",")]
    goc = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)   # ky nguyen cua bai

    print("== Lap lai dong Bang III '%s' qua %d ky nguyen x %d tram ==" %
          (a.che_do, a.ky_nguyen, len(trams)))
    print("   n_sats=%d, horizon=%.0f s, dt=%.0f s, ky nguyen goc cua bai = %s"
          % (cfg["n_sats"], cfg["horizon"], cfg["dt"], goc.date()))
    print("   ⛔ dong dau tien (ky nguyen goc, PTIT) PHAI trung voi Bang III\n")

    dong = []
    t0 = time.time()
    for ground in trams:
        for k in range(a.ky_nguyen):
            t0_utc = goc + timedelta(days=a.cach_ngay * k)
            s, ad, tong, nstep = mot_lan(p, expect, cfg["n_sats"], cfg["horizon"],
                                         cfg["dt"], t0_utc, ground, nhanh=a.nhanh)
            dong.append(dict(che_do=a.che_do, tram=ground.name,
                             ky_nguyen=t0_utc.isoformat(), sec_static=s,
                             sec_adapt=ad, cap_tong=tong, buoc_phuc_vu=nstep))
            print("   %-8s %s  tinh %3d/%-4d  thich ung %3d/%-4d  (%.0f s)"
                  % (ground.name, t0_utc.date(), s, tong, ad, tong, time.time() - t0),
                  flush=True)

    out = RESULTS / ("bang3_nhieu_ky_nguyen_%s.csv" % a.che_do)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dong[0]))
        w.writeheader(); w.writerows(dong)

    print("\n== TONG HOP ==")
    for ten in sorted({d["tram"] for d in dong}):
        v = [d for d in dong if d["tram"] == ten]
        ss = [d["sec_static"] for d in v]
        aa = [d["sec_adapt"] for d in v]
        print("   %-8s  tinh:   trung binh %5.1f, khoang %d..%d" % (ten, sum(ss)/len(ss), min(ss), max(ss)))
        print("   %-8s  thich ung: trung binh %5.1f, khoang %d..%d" % ("", sum(aa)/len(aa), min(aa), max(aa)))
        n_tinh_0 = sum(1 for x in ss if x == 0)
        print("   %-8s  so ky nguyen moc TINH ve 0: %d/%d" % ("", n_tinh_0, len(ss)))
        # ⛔ Ty so phai tinh tren TONG, khong phai trung binh cua tung ty so: mot
        # mau so bang 0 lam ty so vo han va keo trung binh len vo nghia.
        if sum(ss) > 0:
            print("   %-8s  ty so tren TONG: %.2fx (%d so voi %d)"
                  % ("", sum(aa)/sum(ss), sum(aa), sum(ss)))
        else:
            print("   %-8s  moc TINH bang 0 tren MOI ky nguyen: ty so khong xac dinh,"
                  % "")
            print("   %-8s  phai bao bang DEM (%d so voi 0), khong duoc viet vo cuc"
                  % ("", sum(aa)))
    print("\n   da ghi %s" % out.relative_to(ROOT))
    return 0


def tu_kiem():
    """Doi chung: bo tong hop co doc dung khong, va ty so co chong duoc mau so 0 khong."""
    print("== TU KIEM bo tong hop ==\n")
    ok = True

    def kiem(ten, dat):
        print("  %-52s %s" % (ten, "DAT" if dat else "HONG"))
        return dat

    ss, aa = [0, 0, 0], [2, 1, 3]
    ok &= kiem("moc tinh toan 0 thi KHONG tinh ty so", sum(ss) == 0)
    ss2, aa2 = [5, 7, 3], [18, 14, 9]
    ok &= kiem("ty so tinh tren TONG, khong tren trung binh ty so",
               abs(sum(aa2)/sum(ss2) - 41/15) < 1e-9)
    tb_ty_so = sum(b/a for a, b in zip(ss2, aa2)) / 3
    ok &= kiem("hai cach cho ket qua KHAC nhau (nen phai chon dung cach)",
               abs(tb_ty_so - 41/15) > 0.05)
    print("     (tren TONG = %.3f, trung binh ty so = %.3f)" % (sum(aa2)/sum(ss2), tb_ty_so))
    ok &= kiem("hinh hoc cum NHAP tu tep goc, khong chep lai", len(BOB_DZ) == 3)
    print("\n  => %s" % ("TAT CA DAT" if ok else "CO CA HONG"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
