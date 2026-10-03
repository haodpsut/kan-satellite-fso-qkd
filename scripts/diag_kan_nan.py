#!/usr/bin/env python3
"""CHAN DOAN: bao nhieu hat giong bi KAN huan luyen PHAN KY, va bi giau di.

⛔ VI SAO CO TEP NAY. `KANModel.predict` trong train_cluster.py co doan:

    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = self.y_med        # "safety: replace NaN/Inf with training median"

Nghia la khi LBFGS phan ky, du doan cua KAN bi thay bang MOT HANG SO (trung vi tap
huan luyen) va **khong co gi bao ra**. Mot lan huan luyen HONG van cho mot dong so
trong binh thuong, va no di thang vao Bang IV.

Do la lop loi 24 trong so nha: *huan luyen PHAN KY duoc bao nhu mot phep do*. No cung
giai thich thang loi than cua Reviewer 2 diem 2 (do lech chuan cung bac voi gia tri)
va cua Reviewer 4 diem 8 (bat on dang ke): neu mot phan hat giong la BO DU DOAN HANG SO
con phan kia la mo hinh that, thi phuong sai lon la he qua tat yeu, va trung binh cua
hai thu do khong co nghia gi.

⛔ KHONG sua ham predict o day. DO TRUOC, SUA SAU: sua ngay thi mat luon thong tin ban
da cong bo bi anh huong bao nhieu.

    python scripts/diag_kan_nan.py --seeds 20 --device cuda
"""
import argparse
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

so = {"seed": None}
nhat_ky = {}          # seed -> dict(goi, co_nan, o_nan, o_tong)


def ghi(sd):
    return nhat_ky.setdefault(sd, {"goi": 0, "co_nan": 0, "o_nan": 0, "o_tong": 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    fit_goc = TC.KANModel.fit
    pred_goc = TC.KANModel.predict

    def fit_ghi(self, X, y, seed=0):
        so["seed"] = seed
        self._seed = seed
        return fit_goc(self, X, y, seed=seed)

    def pred_ghi(self, X):
        sd = getattr(self, "_seed", so["seed"])
        Xt = self.t.tensor(X, dtype=self.t.float32, device=self.dev)
        with self.t.no_grad():
            out = self.m(Xt).cpu().numpy()
        bad = ~np.isfinite(out)
        g = ghi(sd)
        g["goi"] += 1
        g["o_tong"] += bad.size
        if bad.any():
            g["co_nan"] += 1
            g["o_nan"] += int(bad.sum())
        return pred_goc(self, X)

    TC.KANModel.fit = fit_ghi
    TC.KANModel.predict = pred_ghi

    sys.argv = ["train_cluster.py", "--models", "kan,mlp,linear",
                "--seeds", str(a.seeds), "--device", a.device, "--residual"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            TC.main()
    except SystemExit:
        pass

    print("CHAN DOAN nhanh NaN cua KAN, bo dieu khien cum, %d hat giong\n" % a.seeds)
    print("%5s | %8s %8s | %11s | %s" % ("seed", "lan goi", "lan NaN", "ty le o NaN", "phan quyet"))
    print("-" * 64)
    hong = 0
    for sd in sorted(nhat_ky):
        g = nhat_ky[sd]
        ty = 100.0 * g["o_nan"] / g["o_tong"] if g["o_tong"] else 0.0
        if g["co_nan"]:
            hong += 1
        print("%5d | %8d %8d | %10.1f%% | %s"
              % (sd, g["goi"], g["co_nan"], ty,
                 "⛔ PHAN KY, du doan thanh HANG SO" if g["co_nan"] else "ok"))
    n = len(nhat_ky)
    print("\n  ⭐ %d/%d hat giong co it nhat mot lan roi vao nhanh NaN" % (hong, n))
    if hong:
        print("     Voi nhung hat giong ay, du doan cua KAN KHONG phai mo hinh da hoc,")
        print("     ma la trung vi tap huan luyen. Chung van duoc tinh vao trung binh")
        print("     va do lech chuan cua Bang IV.")

    print("\n---- bang ket qua cua chinh lan chay tren, de doi chieu ----")
    for line in buf.getvalue().splitlines():
        if line.strip() and not line.startswith(("saving", "checkpoint", "description")):
            if "|" not in line or "train_loss" not in line:
                print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
