#!/usr/bin/env python3
"""R2-1: biện minh nguong ro ri lien nguoi dung tau bang SO, va do CHENH DAC TA.

⛔ PHAT HIEN 02/10. Ban thao viet, formulation.tex dong 39:
    "The unkept fraction ... leaks to other users; we cap it at $\\tau{=}0.1$."
nhung MOI duong chay trong ma deu dung mac dinh `leak_max = 0.05`, va khong cho goi
nao truyen gia tri khac. Tuc bai CONG BO mot rang buoc ma ma KHONG thi hanh, va ma
con CHAT GAP DOI. Reviewer 2 diem 1 lai dang hoi hay bien minh cho con so 0,1, tuc
hoi ve mot so chua tung duoc dung.

Day la lop loi "dac ta khac nguon": khong cong nao bat duoc vi khong cong nao doi
chieu con so trong van ban voi con so trong ma.

Tep nay lam hai viec:
  1. QUET tau tren mot dai, do so cap kha thi va tong toc do khoa, de co duong cong
     bien minh thay vi mot khang dinh;
  2. in thang hai cot 0,05 va 0,10 canh nhau, de biet viec sua dac ta co doi ket luan
     nao khong.

    python3 scripts/quet_tau.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_cluster as tc                                       # noqa: E402
from satqkd import (SystemParams, make_expectation, NodeState,   # noqa: E402
                    cluster_key_rate)

TAU = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 1.00]
TAU_MA = 0.05      # gia tri MA dang dung
TAU_BAI = 0.10     # gia tri BAI dang viet


def main():
    g = tc._read_csv(tc.RESULTS / "cluster_global.csv")
    u = tc._read_csv(tc.RESULTS / "cluster_user.csv")
    by = collections.defaultdict(list)
    for r in u:
        by[int(r["cluster_id"])].append(r)
    p = SystemParams()
    ex = make_expectation(p.gh_order)

    print("== R2-1: quet nguong ro ri lien nguoi dung tau ==")
    print("   %d cum, tham so cua chinh ORACLE (khong qua bo dieu khien)\n" % len(g))
    print("   %-7s %10s %10s %14s" % ("tau", "kha thi", "ty le", "tong rate (bit/s)"))
    print("   " + "-" * 46)

    ket = {}
    for tau in TAU:
        nf = ntot = 0
        tong = 0.0
        for gr in g:
            urs = by[int(gr["cluster_id"])]
            if not urs:
                continue
            alice = NodeState(gr["gA"], gr["sA"], gr["wA"])
            bobs = [NodeState(x["g_i"], x["s_i"], x["w_i"]) for x in urs]
            res = cluster_key_rate(gr["mu"], gr["beta_A"],
                                   [x["beta_i"] for x in urs], gr["chi"],
                                   alice, bobs, ex, d_eve=float(urs[0]["d_eve"]),
                                   leak_max=tau)
            nf += res["n_feasible"]
            ntot += len(bobs)
            tong += sum(r for r, f in zip(res["per_rate"], res["per_feasible"]) if f)
        ket[tau] = (nf, ntot, tong)
        danh = ""
        if abs(tau - TAU_MA) < 1e-12:
            danh = "  <== MA dang dung"
        if abs(tau - TAU_BAI) < 1e-12:
            danh = "  <== BAI dang viet"
        print("   %-7.3f %10d %9.1f%% %14.4g%s"
              % (tau, nf, 100.0 * nf / max(ntot, 1), tong, danh))

    a = ket[TAU_MA]
    b = ket[TAU_BAI]
    print("\n== CHENH DAC TA: ma dung %.2f, bai viet %.2f ==" % (TAU_MA, TAU_BAI))
    print("   kha thi : %d  ->  %d   (chenh %+d cap)" % (a[0], b[0], b[0] - a[0]))
    print("   tong rate: %.6g -> %.6g   (chenh %+.3f%%)"
          % (a[2], b[2], 100.0 * (b[2] - a[2]) / a[2] if a[2] else 0.0))
    if a[0] == b[0] and abs(b[2] - a[2]) <= 1e-9 * max(1.0, a[2]):
        print("\n   ⇒ Hai gia tri cho KET QUA GIONG HET. Chenh dac ta la loi VAN BAN,")
        print("     khong doi mot ket luan nao. Sua van ban cho khop ma (hoac nguoc lai)")
        print("     va khai trong thu tra loi.")
    else:
        print("\n   ⛔ Hai gia tri cho ket qua KHAC NHAU. Moi con so trong bai deu phai")
        print("     xac dinh lai theo gia tri DA THUC SU chay, va phai khai ro.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
