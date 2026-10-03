"""Phep thu GHEP CAP cho so sanh trung tam cua bai: KAN so voi MLP tren chi.

⛔ VI SAO CO TEP NAY. Nguoi doc ngoai 03/10/2026 chi ra mot dieu dung va nang: bai
bao cao 0,010+-0,001 so voi 0,018+-0,022 tren 20 hat giong ma KHONG kem phep thu nao.
Tu hai con so do nguoi phan bien tu tinh duoc phep thu KHONG GHEP CAP:

    hieu 0,008 | SE hieu 0,0049 | t ~ 1,62 | p ~ 0,12   => khong phan biet duoc

Ho noi dung: "As written, a referee computes the unpaired version and concludes the
headline accuracy advantage is not established." Nhung cac mo hinh dung CHUNG hat
giong, nen phep thu dung la GHEP CAP, va no co the cho ket luan khac han.

Tep nay chay ba thu tren cung mot bo du lieu:
  1. Wilcoxon ghep cap (hai phia), chinh xac neu n nho
  2. do lon hieu ung ghep cap: Cliff delta va ty le thang/thua/hoa
  3. ty so DO PHAN TAN (sd_mlp / sd_kan), cai ma nguoi doc ngoai noi moi la ket qua song

⛔ LUAT: in ra ket qua DU NO NGHIENG VE PHIA NAO. Neu ghep cap cung khong co y nghia
thong ke thi phai viet dung nhu vay vao bai, va chuyen tuyen bo chinh sang do phan tan.

    python3 scripts/phep_thu_ghep_cap.py
    python3 scripts/phep_thu_ghep_cap.py --tu-kiem    # doi chung tren du lieu BIET TRUOC
"""
import argparse
import csv
import io
import itertools
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "results", "cluster_per_seed.csv")


def wilcoxon_chinh_xac(d):
    """Wilcoxon signed-rank HAI PHIA, tinh chinh xac bang duyet het 2^n dau.

    Dung cach duyet het thay vi xap xi chuan vi n = 20 thi xap xi con tho, va vi
    bai da tung bi bat loi "p-value phong hai bac" o cho khac. 2^20 = 1,05 trieu,
    chay het duoi mot giay.
    """
    d = [x for x in d if x != 0.0]
    n = len(d)
    if n == 0:
        return float("nan"), 0, float("nan")
    # hang cua |d|, xu ly hoa bang hang trung binh
    idx = sorted(range(n), key=lambda i: abs(d[i]))
    hang = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[idx[j + 1]]) == abs(d[idx[i]]):
            j += 1
        tb = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            hang[idx[k]] = tb
        i = j + 1
    W_duong = sum(h for h, x in zip(hang, d) if x > 0)
    tong = sum(hang)
    T = min(W_duong, tong - W_duong)
    if n > 22:            # duyet het se qua lau, dung xap xi chuan co hieu chinh
        mu = tong / 2.0
        sd = math.sqrt(sum(h * h for h in hang) / 4.0)
        z = (T - mu + 0.5) / sd
        p = math.erfc(abs(z) / math.sqrt(2.0))
        return p, n, T
    # duyet het moi to hop dau
    dem = 0
    tongs = 2 ** n
    for mask in range(tongs):
        s = 0.0
        m = mask
        for k in range(n):
            if m & 1:
                s += hang[k]
            m >>= 1
        if min(s, tong - s) <= T + 1e-12:
            dem += 1
    return dem / tongs, n, T


def cliff_delta(a, b):
    """Cliff delta ghep cap khong dung duoc; day la ban khong ghep cap, de tham khao."""
    hon = sum(1 for x, y in itertools.product(a, b) if x < y)   # a tot hon (MAE nho hon)
    kem = sum(1 for x, y in itertools.product(a, b) if x > y)
    return (hon - kem) / float(len(a) * len(b))


def doc():
    if not os.path.exists(CSV):
        sys.exit("⛔ chua co %s. Chay truoc:\n"
                 "   python3 scripts/train_cluster.py --models kan,mlp,linear "
                 "--seeds 20 --residual --bo-muc-tieu-suy-bien --device cuda"
                 % os.path.relpath(CSV, ROOT))
    r = list(csv.DictReader(io.open(CSV, encoding="utf-8")))
    d = {}
    for x in r:
        d.setdefault(x["model"], {})[int(x["seed"])] = x
    return d


def sd(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def bao_cao(ten_a, ten_b, a, b, hat):
    print("\n== %s so voi %s tren chi, %d hat giong CHUNG ==" % (ten_a, ten_b, len(hat)))
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa, sb = sd(a), sd(b)
    print("   %-10s trung binh %.4f, do lech %.4f" % (ten_a, ma, sa))
    print("   %-10s trung binh %.4f, do lech %.4f" % (ten_b, mb, sb))
    print("   ty so DO PHAN TAN (%s/%s): %.1fx" % (ten_b, ten_a, (sb / sa) if sa > 0 else float("inf")))
    hieu = [y - x for x, y in zip(a, b)]          # duong = b te hon = a tot hon
    thang = sum(1 for h in hieu if h > 0)
    thua = sum(1 for h in hieu if h < 0)
    hoa = sum(1 for h in hieu if h == 0)
    print("   theo cap: %s tot hon o %d/%d hat, kem hon %d, hoa %d"
          % (ten_a, thang, len(hieu), thua, hoa))
    p, n_hd, T = wilcoxon_chinh_xac(hieu)
    print("   Wilcoxon ghep cap (chinh xac, hai phia): p = %.5g  (n hieu khac 0 = %d, T = %g)"
          % (p, n_hd, T))
    print("   Cliff delta (khong ghep cap, tham khao): %.3f" % cliff_delta(a, b))
    # Phep thu KHONG ghep cap, chi de doi chieu voi con so nguoi doc ngoai da tu tinh.
    # Dung do lech chuan MAU (chia n-1) vi do la cai di vao cong thuc Welch.
    sa1 = (sum((x - ma) ** 2 for x in a) / (len(a) - 1)) ** 0.5
    sb1 = (sum((x - mb) ** 2 for x in b) / (len(b) - 1)) ** 0.5
    se = math.sqrt(sa1 ** 2 / len(a) + sb1 ** 2 / len(b))
    t = (mb - ma) / se if se > 0 else float("nan")
    print("   (de doi chieu) t KHONG ghep cap = %.2f, dung con so nguoi doc ngoai tinh" % t)
    return dict(p=p, thang=thang, n=len(hieu), ty_so_pt=(sb / sa) if sa > 0 else float("inf"),
                mean_a=ma, mean_b=mb, sd_a=sa, sd_b=sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    if ap.parse_args().tu_kiem:
        return tu_kiem()

    d = doc()
    if "kan" not in d or "mlp" not in d:
        sys.exit("⛔ thieu dong cho kan hoac mlp trong %s" % CSV)
    hat = sorted(set(d["kan"]) & set(d["mlp"]))
    if len(hat) < 2:
        sys.exit("⛔ chi co %d hat giong chung, khong chay duoc phep thu" % len(hat))
    # ⛔ Loai hat giong co o PHAN KY: lop loi 24, mot lan huan luyen phan ky khong
    # phai mot phep do, nen khong duoc dua vao phep thu.
    bo = [s for s in hat if float(d["kan"][s]["n_div"]) > 0 or float(d["mlp"][s]["n_div"]) > 0]
    if bo:
        print("   chu y: %d/%d hat giong co o phan ky, VAN giu trong phep thu nhung "
              "bao rieng: %s" % (len(bo), len(hat), bo))
    a = [float(d["kan"][s]["mae_chi"]) for s in hat]
    b = [float(d["mlp"][s]["mae_chi"]) for s in hat]
    kq = bao_cao("KAN", "MLP+resid", a, b, hat)

    if "linear" in d:
        hat2 = sorted(set(d["kan"]) & set(d["linear"]))
        bao_cao("KAN", "Linear",
                [float(d["kan"][s]["mae_chi"]) for s in hat2],
                [float(d["linear"][s]["mae_chi"]) for s in hat2], hat2)

    if any("us_pred" in v for v in d["kan"].values()):
        print("\n== do tre M4 (us moi lan du doan) ==")
        for k in sorted(d):
            v = [float(x["us_pred"]) for x in d[k].values() if x.get("us_pred")]
            if v:
                print("   %-10s %8.1f +- %.1f us" % (k, sum(v) / len(v), sd(v)))

    print("\n== KET LUAN PHAI VIET VAO BAI ==")
    if kq["p"] < 0.05:
        print("   Ghep cap CO y nghia (p = %.4g): duoc phep neu loi the do chinh xac,"
              % kq["p"])
        print("   nhung VAN phai bao ca phep thu khong ghep cap vi phan bien se tu tinh no.")
    else:
        print("   Ghep cap KHONG co y nghia (p = %.4g). Phai bo tuyen bo loi the do"
              % kq["p"])
        print("   chinh xac khoi abstract va chuyen sang ty so do phan tan %.0fx."
              % kq["ty_so_pt"])
    return 0


def tu_kiem():
    """Doi chung: chay bo thu tren du lieu BIET TRUOC dap an."""
    print("== TU KIEM phep thu ghep cap ==\n")
    ok = True

    def thu(ten, hieu, p_mong, sai_so=0.02):
        p, n, T = wilcoxon_chinh_xac(hieu)
        dat = abs(p - p_mong) <= sai_so
        print("  %-46s p = %.5f (mong %.3f)  %s" % (ten, p, p_mong, "DAT" if dat else "HONG"))
        return dat

    # 1. moi cap deu cung dau, n = 10 => p chinh xac = 2/2^10 = 0,001953
    ok &= thu("10 cap cung dau (mong 2/2^10)", [0.1] * 10, 2 / 1024.0, 1e-6)
    # 2. doi xung hoan toan => p = 1
    ok &= thu("6 cap doi xung (mong p = 1)", [1, -1, 2, -2, 3, -3], 1.0, 1e-9)
    # 3. n = 20 cung dau => 2/2^20
    ok &= thu("20 cap cung dau (mong 2/2^20)", [0.5] * 20, 2 / 1048576.0, 1e-9)
    # 4. ca KHONG the bac bo: 1 cap
    p, n, T = wilcoxon_chinh_xac([0.3])
    print("  %-46s p = %.5f (mong 1.0)  %s" % ("1 cap duy nhat", p, "DAT" if abs(p - 1.0) < 1e-9 else "HONG"))
    ok &= abs(p - 1.0) < 1e-9
    # 5. doi chung AM: hieu toan 0 thi khong co gi de thu
    p, n, T = wilcoxon_chinh_xac([0.0, 0.0, 0.0])
    print("  %-46s n = %d  %s" % ("hieu toan 0 thi tra n = 0", n, "DAT" if n == 0 else "HONG"))
    ok &= (n == 0)
    print("\n  => %s" % ("TAT CA DAT" if ok else "CO CA HONG"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
