#!/usr/bin/env python3
"""Ban thao TCOM da du tu cach gui RA NGOAI doc lai chua?

⛔ VI SAO CO TEP NAY. Vong doc ngoai 03/10/2026 tra MAJOR voi sau viec. Cau hoi
truoc khi gui lai khong phai "toi co ban moi khong" ma la "sau viec do, viec nao
DA CO TRONG BAN THAO va viec nao chua". Hai lop loi phai chan:

  (A) Khai mot thay doi ma ban thao KHONG co. Da xay ra that o bai IoT-J: cover
      letter khai ba thay doi, mot trong ba khong he co trong ban thao, va nguoi
      doc ngoai tim ra bang MOT lan tim chuoi. Nen o day moi muc deu phai doi mot
      BANG CHUNG la chuoi ky tu co that trong PDF da dung.
  (B) Gui di khi hai muc NANG NHAT con thieu so do, roi nhan lai dung hai muc do.
      Cong nay phai noi thang "chua nen gui" trong truong hop ay, chu khong cho
      diem thanh phan roi de nguoi doc tu suy.

    python3 scripts/kiem_san_sang_doc_ngoai.py
    python3 scripts/kiem_san_sang_doc_ngoai.py --tu-kiem
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF = os.path.join(ROOT, "paper", "main.pdf")

# Moi muc: (ma, muc do, mo ta, bang chung phai co trong PDF, bang chung KHONG duoc con)
MUC = [
    ("V1", "CHAN",
     "Phep thu ghep cap KAN vs MLP tren chi, kem do lon hieu ung",
     ["signed-rank", "paired"], []),
    ("V2", "nhe",
     "Abstract tach chi (cum) khoi luat dong cho beta* (don lien ket)",
     ["in the cluster setting", "in the single-link setting"], []),
    ("V3", "nhe",
     "Bang III bo ty so vo cuc, thay bang dem",
     # ⛔ HAI lan tu sua cho nay, ca hai deu do doi chung bat:
     # (1) tim chuoi nguon LaTeX "$\\infty$" thi khong bao gio khop, vi PDF ra ky tu
     #     "\u221e". Phai tim ky tu da dung.
     # (2) cam ky tu "\u221e" o BAT KY dau lai bao nham: ma gia khoi tao "V* <- -\u221e"
     #     la cach dung toan hoc hop le va khong lien quan ty so Bang III. Nen cam
     #     theo NGU CANH ty so, khong cam ca ky tu.
     # ⛔ lan sua thu ba: "[^.]" khong di qua duoc dau cham thap phan cua "3.07"
     # nen mau khong bao gio khop. Dung ".{0,120}".
     ["survival rather than operation"],
     [r"\u00d7\s*,?\s*\u221e", r"gains.{0,120}\u221e"]),
    ("V4", "nhe",
     "Cot do tre co trong bang",
     ["116.7"], []),
    ("V5", "CHAN",
     "Lap lai dong Bang III qua nhieu ky nguyen TLE, bao trung binh va khoang",
     ["TLE epochs"], []),
    ("V6", "nhe",
     "Trich va dinh vi van lieu so sanh KAN-MLP cung so tham so",
     ["fairer", "symbolic formula representation"], []),
    ("V7", "nhe",
     "Noi ro bao nhieu truc that su tach duoc hai ho",
     ["$M_2$ is dead", "M2 is dead"], []),
    ("V8", "nhe",
     "Noi ro bo dieu khien hoc duoc mua gi so voi anh xa tuyen tinh",
     ["linear map is narrower", "buys over the nine-parameter linear map"], []),
    ("V9", "nhe",
     "R2 toan mang va R2 cua luat trich duoc tach bach",
     ["full-network", "post-extraction"], []),
]


def chu(pdf):
    """Chu cua PDF, da noi lai cac tu bi ngat dong co gach noi.

    ⛔ Khong dung -layout: o bai IoT-J, -layout tron HAI COT vao cung mot dong va
    phep dem tren no ra 576 tu cho mot abstract 244 tu. O day ta chi tim chuoi nen
    can dong chay dung thu tu doc, tuc KHONG -layout.
    """
    t = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    t = re.sub(r"-\s*\n\s*", "", t)          # noi tu bi ngat dong
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    ap.add_argument("--pdf", default=PDF)
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()

    if not os.path.exists(a.pdf):
        sys.exit("⛔ chua co %s" % a.pdf)
    t = chu(a.pdf).lower()

    print("== Ban thao TCOM da du tu cach gui ra ngoai doc lai chua? ==\n")
    print("   %-4s %-6s %-58s %s" % ("ma", "muc do", "viec", "trang thai"))
    print("   " + "-" * 92)
    thieu_chan, thieu_nhe = [], []
    for ma, muc, mota, can, cam in MUC:
        co = any(c.lower() in t for c in can)
        # muc "cam" la bieu thuc chinh quy, de cam duoc theo NGU CANH chu khong
        # cam ca mot ky tu dung o cho khac hoan toan hop le
        du = all(re.search(c.lower(), t) is None for c in cam)
        dat = co and du
        if not dat:
            (thieu_chan if muc == "CHAN" else thieu_nhe).append((ma, mota))
        print("   %-4s %-6s %-58s %s" % (ma, muc, mota[:58], "CO" if dat else "CHUA"))

    print("\n   " + "-" * 92)
    if thieu_chan:
        print("\n   => CHUA NEN GUI. %d muc CHAN con thieu so do:" % len(thieu_chan))
        for ma, m in thieu_chan:
            print("      %s  %s" % (ma, m))
        print("\n      Gui di luc nay thi vong doc sau se tra lai dung may muc nay,")
        print("      vi chung la hai viec dau tien trong danh sach cua vong truoc.")
    elif thieu_nhe:
        print("\n   => GUI DUOC, nhung %d muc nhe con thieu:" % len(thieu_nhe))
        for ma, m in thieu_nhe:
            print("      %s  %s" % (ma, m))
    else:
        print("\n   => DU TU CACH GUI: ca %d muc deu co bang chung trong ban thao." % len(MUC))
    return 1 if thieu_chan else 0


def tu_kiem():
    """Doi chung: cong phai bat duoc ca KHAI KHONG co THAT."""
    import tempfile
    print("== TU KIEM cong san sang ==\n")
    ok = True
    d = tempfile.mkdtemp()

    def lam_pdf(noi_dung, ten):
        tex = os.path.join(d, ten + ".tex")
        open(tex, "w").write(
            "\\documentclass{article}\\begin{document}\n" + noi_dung + "\n\\end{document}\n")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", d, tex],
                       capture_output=True)
        return os.path.join(d, ten + ".pdf")

    # 1. ban KHONG co gi: moi muc phai bao CHUA, va phai ket luan CHUA NEN GUI
    p1 = lam_pdf("Mot ban thao rong khong noi gi ca.", "rong")
    r = subprocess.run([sys.executable, __file__, "--pdf", p1], capture_output=True, text=True)
    bat = "CHUA NEN GUI" in r.stdout and r.returncode == 1
    print("  %-56s %s" % ("ban rong: phai bao CHUA NEN GUI", "DAT" if bat else "HONG"))
    ok = ok and bat

    # 2. ban co DU bang chung cua hai muc CHAN: khong duoc bao chan nua
    p2 = lam_pdf("We report a paired signed-rank test. We repeat over TLE epochs.", "chan")
    r = subprocess.run([sys.executable, __file__, "--pdf", p2], capture_output=True, text=True)
    het = "CHUA NEN GUI" not in r.stdout
    print("  %-56s %s" % ("co bang chung 2 muc CHAN: thoi bao chan", "DAT" if het else "HONG"))
    ok = ok and het

    # 3. ⛔ doi chung AM quan trong nhat: chuoi BI CAM van con thi phai bao CHUA
    p3 = lam_pdf("Secure-time gains: $3.07\\times$, $2.50\\times$, $3.60\\times$, "
                 "$\\infty$. Survival rather than operation.", "cam")
    r = subprocess.run([sys.executable, __file__, "--pdf", p3], capture_output=True, text=True)
    v3 = [l for l in r.stdout.splitlines() if l.strip().startswith("V3")]
    bat3 = v3 and v3[0].strip().endswith("CHUA")
    print("  %-56s %s" % ("con ky hieu vo cuc: V3 phai bao CHUA", "DAT" if bat3 else "HONG"))
    ok = ok and bat3

    # 4. ⛔ doi chung BAO NHAM: "-\u221e" trong ma gia la hop le, KHONG duoc bao CHUA.
    #    Chinh ca nay da lam cong bao nham mot lan ngay 03/10.
    p4 = lam_pdf("We set $V^\\star \\gets -\\infty$ in the pseudocode. "
                 "Survival rather than operation.", "hople")
    r = subprocess.run([sys.executable, __file__, "--pdf", p4], capture_output=True, text=True)
    v3b = [l for l in r.stdout.splitlines() if l.strip().startswith("V3")]
    im = bool(v3b) and v3b[0].strip().endswith("CO")
    print("  %-56s %s" % ("'-inf' trong ma gia: V3 KHONG duoc bao nham", "DAT" if im else "HONG"))
    ok = ok and im

    print("\n  => %s" % ("TAT CA DAT" if ok else "CO CA HONG"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
