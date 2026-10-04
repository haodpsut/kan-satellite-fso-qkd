#!/usr/bin/env python3
"""Cong DINH DANG cua noi nhan (TCOM). Chay TRUOC moi lan dung goi nop.

⛔ VI SAO CO TEP NAY. Ngay 03/10/2026 cung mot lop loi bi bat HAI lan trong mot buoi,
o HAI bai khac nhau: so do CO SAN hoac de dang do duoc, nhung khong ai dat nguong lay
tu huong dan cua NOI NHAN, nen ban thao troi ra ngoai han ma moi cong noi bo van xanh
va CA HAI vong doc ngoai deu khong thay (ho doc noi dung khoa hoc, khong mo huong dan
tac gia ra doi chieu).

Ban R1 cua bai nay khi bi bat: abstract 306 tu (han 75-200) va ban thao 15 trang
(han 13). Ban v1 da nop la 219 tu / 12 trang.

Nguyen van han, tai comsoc.org/publications/journals/ieee-tcom/policies-and-guidelines:
  "It is essential that each manuscript be accompanied by a 75- to 200-word abstract
   clearly outlining the scope and contributions of the paper"
  "Transactions Papers should be concisely written and may not exceed 13 double-column
   single-spaced pages (10-point font size, 1-inch margin on all sides)"
  "All papers submitted after 1 January 2020 that are accepted for publication are
   subject to a mandatory page charge of US$220 for each Transactions page exceeding
   ten printed pages."

    python3 scripts/kiem_dinh_dang_tcom.py
    python3 scripts/kiem_dinh_dang_tcom.py --tu-kiem
"""
import argparse
import glob
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper")

ABS_LO, ABS_HI = 75, 200          # nguyen van huong dan TCOM
TRANG_TOI_DA = 13                 # nguyen van huong dan TCOM
PHI_TU_TRANG = 10                 # phi bat dau tu trang thu 11
PHI_MOI_TRANG = 220               # USD

loi = []


def kiem(ok, ten, ct=""):
    print("  %s %s%s" % ("DAT " if ok else "HONG", ten, ("  [" + ct + "]") if ct else ""))
    if not ok:
        loi.append(ten)
    return ok


def nap_macro():
    """Gia tri macro, de \\rAbc khong bi dem thanh mot tu bien mat hoac mot tu thua."""
    mac = {}
    for f in glob.glob(os.path.join(PAPER, "**", "*.tex"), recursive=True):
        try:
            s = io.open(f, encoding="utf-8").read()
        except Exception:
            continue
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", s):
            mac[m.group(1)] = m.group(2)
    return mac


def dem_tu_abstract(path, mac=None):
    """Dem tu cua abstract tren NGUON, sau khi khai trien macro.

    ⛔ Hai cai bay da gap: (1) '\\%' bi tach thanh mot tu rieng neu khong dan lien,
    (2) macro bi xoa sach thi mot con so bien mat khoi phep dem. Ca hai deu lam
    bo dem noi lao theo huong CO LOI cho minh, tuc huong nguy hiem.
    """
    s = io.open(path, encoding="utf-8").read()
    s = re.sub(r"(?m)^\s*%.*$", "", s)                  # bo dong chu thich
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    # ⛔ 04/10: cho nay TRUOC day co nhanh du phong `else s`, tuc thieu moi truong
    # abstract thi lang le dem CA TEP. Chinh nhanh do da che mot loi that: ban R1
    # mat han \begin{abstract} nen PDF khong co dau chay "Abstract—", va cong van
    # bao so tu binh thuong. Nguoi doc ngoai tim ra bang mot lan tim chuoi.
    # Khong co moi truong thi KHONG phai la "dem duoc 0 tu", ma la KHONG DO DUOC.
    if not m:
        raise LookupError("khong tim thay \\begin{abstract}...\\end{abstract} trong %s"
                          % os.path.basename(path))
    t = m.group(1)
    t = t.replace(r"\%", "%").replace(r"\,", "")
    t = re.sub(r"\\(\w+)", lambda g: (mac or {}).get(g.group(1), " "), t)
    return len(re.sub(r"[${}~\\]", "", t).split())


def so_trang(pdf):
    if not os.path.exists(pdf):
        return 0
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    ap.add_argument("--pdf", default=os.path.join(PAPER, "main.pdf"))
    ap.add_argument("--abstract", default=os.path.join(PAPER, "sections", "abstract.tex"))
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()

    print("== Cong dinh dang TCOM (han lay nguyen van tu huong dan tac gia) ==\n")
    mac = nap_macro()

    try:
        n = dem_tu_abstract(a.abstract, mac)
        kiem(ABS_LO <= n <= ABS_HI, "T1 abstract %d-%d tu" % (ABS_LO, ABS_HI),
             "%d tu%s" % (n, "" if ABS_LO <= n <= ABS_HI else
                          " (VUOT %d)" % (n - ABS_HI if n > ABS_HI else ABS_LO - n)))
    except LookupError as e:
        kiem(False, "T1 abstract %d-%d tu" % (ABS_LO, ABS_HI), str(e))

    tr = so_trang(a.pdf)
    kiem(0 < tr <= TRANG_TOI_DA, "T2 ban thao khong qua %d trang" % TRANG_TOI_DA,
         "%d trang%s" % (tr, "" if 0 < tr <= TRANG_TOI_DA else
                         " (VUOT %d)" % (tr - TRANG_TOI_DA)))

    # ---- T3 dau chay "Abstract—" phai co trong PDF ----
    # Dem tren NGUON khong du: moi truong co the co ma lop tai lieu van khong in dau
    # chay. Phai doc ban DA DUNG, dung thu nguoi bien tap nhin thay.
    if os.path.exists(a.pdf):
        t3 = subprocess.run(["pdftotext", a.pdf, "-"], capture_output=True, text=True).stdout
        kiem("Abstract" in t3 and "Index Terms" in t3,
             "T3 PDF co dau chay Abstract va Index Terms",
             "Abstract %d lan, Index Terms %d lan" % (t3.count("Abstract"), t3.count("Index Terms")))

    # ⛔ KHONG noi nguong 13 cho no im: nguong do la cua tap chi, phai giu nguyen.
    # Chi GHI THEM quyet dinh cua tac gia ben canh, de lan bao HONG van co nghia.
    if tr > TRANG_TOI_DA:
        print("       ghi chu: Hao chot tran 15 trang (quyet dinh 03/10/2026). Ban nay %d "
              "trang, tuc trong tran do nhung VAN vuot han %d cua tap chi."
              % (tr, TRANG_TOI_DA))
    # Khong phai cong: thong bao tien, de nguoi quyet dinh biet gia truoc khi nop.
    if tr:
        vuot = max(0, tr - PHI_TU_TRANG)
        print("       phi vuot trang du tinh: %d trang x %d USD = %d USD"
              % (vuot, PHI_MOI_TRANG, vuot * PHI_MOI_TRANG))

    print("\n=> %s (%d loi)" % ("DAT" if not loi else "CHUA DAT", len(loi)))
    for x in loi:
        print("   loi: %s" % x)
    return 0 if not loi else 1


def tu_kiem():
    """Doi chung: bo dem phai dung tren cac ca BIET TRUOC dap an."""
    print("== TU KIEM bo dem abstract ==\n")
    import tempfile
    ok = True

    def thu(ten, noi_dung, mong, mac=None):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "a.tex")
        io.open(p, "w", encoding="utf-8").write(noi_dung)
        n = dem_tu_abstract(p, mac)
        dat = n == mong
        print("  %-52s %d (mong %d)  %s" % (ten, n, mong, "DAT" if dat else "HONG"))
        return dat

    ok &= thu("nam tu tran", "One two three four five.", 5)
    ok &= thu("dau phan tram dinh lien, khong tach thanh tu rieng",
              r"Gain is $4.1\%$ here.", 4)
    ok &= thu("macro duoc khai trien thanh gia tri, khong bien mat",
              r"Gain is $\rG\%$ here.", 4, {"rG": "5.8"})
    ok &= thu("dong chu thich % khong duoc dem",
              "% day la ghi chu dai dong khong duoc tinh\nOne two three.", 3)
    ok &= thu("chi dem phan trong begin/end abstract",
              r"Bo cai nay. \begin{abstract}One two three.\end{abstract} Va bo cai nay nua.", 3)
    # doi chung AM: bo dem KHONG duoc tra dung cho mot chuoi sai so tu
    d = __import__("tempfile").mkdtemp()
    p = os.path.join(d, "b.tex")
    io.open(p, "w", encoding="utf-8").write("One two three.")
    n = dem_tu_abstract(p)
    print("  %-52s %s" % ("doi chung AM: 3 tu KHONG duoc dem ra 4",
                          "DAT" if n != 4 else "HONG"))
    ok &= (n != 4)
    print("\n  => %s" % ("TAT CA DAT" if ok else "CO CA HONG"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
