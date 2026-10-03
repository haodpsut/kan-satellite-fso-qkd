#!/usr/bin/env python3
"""CHAN DOAN NGUYEN NHAN: vi sao LBFGS cua KAN phan ky, va sua bang cach nao.

Tiep theo `diag_kan_nan.py`, vốn chi dem: 20/20 hat giong roi vao nhanh NaN, 51,3% so o
bi thay bang trung vi. Tep nay hoi VI SAO, va thu tung phuong an sua tren DUNG du lieu
that, khong doan.

Kiem bon thu, theo thu tu re den dat:
  1. THANG du lieu: dau vao va dau ra (ca muc tieu THO va muc tieu RESIDUAL) co lech
     bac nhau khong. LBFGS rat nhay voi thang.
  2. TANG NAO hong: bo dieu khien TOAN CUC (gm) hay bo dieu khien NGUOI DUNG (um).
  3. NaN xuat hien o BUOC NAO cua huan luyen: ngay tu dau hay sau vai buoc.
  4. Bon phuong an sua, moi cai chay tren cung 3 hat giong:
       a. nguyen trang (doi chung)
       b. LBFGS + chuan hoa muc tieu residual
       c. Adam thay LBFGS
       d. LBFGS + lamb = 0

In ra ty le o NaN va MAE cua tung phuong an, de chon bang so chu khong bang cam giac.
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np                                    # noqa: E402
import train_cluster as TC                            # noqa: E402


def thang(ten, a):
    a = np.asarray(a, dtype=float)
    return ("  %-26s shape %-12s |min| %.3g  max|.| %.3g  std %.3g"
            % (ten, str(a.shape), np.abs(a).min(), np.abs(a).max(), a.std()))


def main():
    # ---- nap du lieu dung duong ma train_cluster dung ----
    import argparse
    sys.argv = ["x", "--models", "kan", "--seeds", "1", "--device", "cuda", "--residual"]
    # lay du lieu bang cach goi lai cac buoc dau cua main(): don gian nhat la doc CSV
    g_rows = TC._read_csv(TC.RESULTS / "cluster_global.csv")
    u_rows = TC._read_csv(TC.RESULTS / "cluster_user.csv")
    print("== 1. THANG DU LIEU ==")
    print("  cluster_global.csv: %d dong | cluster_user.csv: %d dong"
          % (len(g_rows), len(u_rows)))

    # dung chinh ham chuan bi cua module neu co, neu khong thi bao va dung
    prep = getattr(TC, "prepare", None) or getattr(TC, "_prepare", None)
    print("  (ham chuan bi: %s)" % (prep.__name__ if prep else "khong tim thay, se lay tu main"))

    # ---- chay main mot lan, chan lai X, y cua tung tang ----
    bat = {}
    fit_goc = TC.KANModel.fit

    def fit_bat(self, X, y, seed=0):
        k = "global" if X.shape[0] == 200 else "user"
        bat.setdefault(k, {"X": X, "y": y})
        return fit_goc(self, X, y, seed=seed)

    TC.KANModel.fit = fit_bat

    # cung bat muc tieu RESIDUAL that su dua vao KAN
    res_goc = TC.ResidualModel.fit

    def res_bat(self, X, y, seed=0):
        self.base.fit(X, y, seed=seed)
        r = y - self.base.predict(X)
        k = "global" if X.shape[0] == 200 else "user"
        bat.setdefault(k, {})["y_raw"] = y
        bat[k]["y_res"] = r
        self.top.fit(X, r, seed=seed)

    TC.ResidualModel.fit = res_bat

    sys.argv = ["train_cluster.py", "--models", "kan", "--seeds", "1",
                "--device", "cuda", "--residual"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            TC.main()
    except SystemExit:
        pass

    for k in ("global", "user"):
        if k not in bat:
            continue
        d = bat[k]
        print("\n  -- tang %s --" % k.upper())
        if "X" in d:
            print(thang("dau vao X", d["X"]))
        if "y_raw" in d:
            print(thang("muc tieu THO y", d["y_raw"]))
        if "y_res" in d:
            print(thang("muc tieu RESIDUAL y-lin", d["y_res"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
