#!/usr/bin/env python3
"""R3-1: quet he so trich cua BEAM SPLITTER (eta) de bien minh bang so.

Phan bien 3 diem 1 hoi hai dieu: (a) URA + BSA co du bao quat cho Eve khong, va
(b) can chi tiet hon ve BSA, cu the la "Eve's channel realization details,
transmittance capacity of the beam-splitter".

Ve (b) do duoc: ban thao co dinh eta_leak = 1,5% va m_min = 0,5% ma khong quet.
Tep nay quet eta tren mot dai, do so cap van kha thi, de bai noi duoc:
"nguong phat hien cua ta bat duoc moi beam splitter trich tu X% tro len".

    python3 scripts/quet_eta.py
"""
import collections, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cluster as tc                                        # noqa: E402
from satqkd import (SystemParams, make_expectation, NodeState,    # noqa: E402
                    cluster_key_rate)

ETA = [0.0100, 0.0110, 0.0115, 0.0120, 0.0125, 0.0130, 0.0135, 0.0140, 0.0145, 0.0150]
ETA_BAI = 0.015


def main():
    g = tc._read_csv(tc.RESULTS / "cluster_global.csv")
    u = tc._read_csv(tc.RESULTS / "cluster_user.csv")
    by = collections.defaultdict(list)
    for r in u:
        by[int(r["cluster_id"])].append(r)
    p = SystemParams(); ex = make_expectation(p.gh_order)
    print("== R3-1: quet he so trich cua beam splitter (eta_leak) ==")
    print("   %d cum, tham so oracle, nguong phat hien m_min = 0,5%%\n" % len(g))
    print("   %-8s %10s %9s %14s" % ("eta", "kha thi", "ty le", "tong rate"))
    print("   " + "-" * 45)
    for eta in ETA:
        nf = ntot = 0; tong = 0.0
        for gr in g:
            urs = by[int(gr["cluster_id"])]
            if not urs: continue
            alice = NodeState(gr["gA"], gr["sA"], gr["wA"])
            bobs = [NodeState(x["g_i"], x["s_i"], x["w_i"]) for x in urs]
            res = cluster_key_rate(gr["mu"], gr["beta_A"],
                                   [x["beta_i"] for x in urs], gr["chi"],
                                   alice, bobs, ex, d_eve=float(urs[0]["d_eve"]),
                                   bsa_split=eta)
            nf += res["n_feasible"]; ntot += len(bobs)
            tong += sum(r for r, f in zip(res["per_rate"], res["per_feasible"]) if f)
        danh = "  <== gia tri bai dung" if abs(eta - ETA_BAI) < 1e-12 else ""
        print("   %-8.4f %10d %8.1f%% %14.4g%s"
              % (eta, nf, 100.0 * nf / max(ntot, 1), tong, danh))
    print("\n   Doc: eta cang NHO thi beam splitter cang kin dao va cang kho phat hien,")
    print("   nen so cap kha thi (tuc so cap ta DAM BAO phat hien duoc) giam. Nguong")
    print("   eta nho nhat con giu duoc kha thi chinh la DO NHAY phat hien cua he.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
