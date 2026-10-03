#!/usr/bin/env python3
"""QA goi nop TCOM R1: kiem goi truoc khi dua cho nguoi doc ngoai.

Tam cong, moi cong tra loi mot cau hoi co the HONG that:
  Q1 goi co du bon thu khong, va co lan ghi chu noi bo khong
  Q2 moi diem phan bien co dung MOT cau tra loi khong, dem tu NGUON GOC
  Q3 ban to sang co thuc su to sang khong, hay chi la ban sach doi ten
  Q4 ban thao va thu tra loi co dung CUNG mot con so khong
  Q5 FRAME co giu khong: tieu de va danh sach dong gop
  Q6 moi nhan float dan trong thu co TON TAI trong ban thao khong
  Q7 goi co TU DUNG LAI duoc khong (giai nen, xoa PDF, dich lai)
  Q8 kho git sach va da day chua

--tu-kiem: tiem loi vao va kiem cong co bat khong.
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOI = os.path.join(ROOT, "submit-R1")
PAPER = os.path.join(HERE, "paper")
SO = os.path.join(ROOT, "reviews", "so-nhan-xet.csv")
NGUON_GOC = os.path.join(ROOT, "reviews", "VONG-1-NGUYEN-VAN-decision-email.md")

loi = []


def kiem(ok, ten, ct=""):
    print("  %s %s%s" % ("DAT " if ok else "HONG", ten, ("  [" + ct + "]") if ct else ""))
    if not ok:
        loi.append(ten)
    return ok


def chu(p):
    """Boc chu tu PDF theo kieu giu bo cuc."""
    return subprocess.run(["pdftotext", "-layout", p, "-"],
                          capture_output=True, text=True).stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()
    if a.tu_kiem:
        return tu_kiem()

    print("== QA goi nop TCOM-TPS-26-1667 R1 ==\n")
    rows = list(csv.DictReader(io.open(SO, encoding="utf-8")))

    # ---- Q1 thanh phan goi ----
    can = ["manuscript-R1.pdf", "manuscript-R1-highlighted.pdf",
           "response-to-reviewers-R1.pdf", "goi-nop-R1.zip"]
    thieu = [f for f in can if not os.path.exists(os.path.join(GOI, f))]
    kiem(not thieu, "Q1a goi co du 4 thanh phan", "thieu " + ", ".join(thieu) if thieu else "")
    ghi_chu = []
    for dp, _, fs in os.walk(GOI):
        for f in fs:
            if os.path.splitext(f)[1].lower() in (".md", ".txt", ".org"):
                ghi_chu.append(os.path.relpath(os.path.join(dp, f), GOI))
    kiem(not ghi_chu, "Q1b khong lan ghi chu noi bo",
         ", ".join(ghi_chu) if ghi_chu else "")

    # ---- Q2 point-to-point, dem tu NGUON GOC ----
    src = io.open(NGUON_GOC, encoding="utf-8").read()
    neo = {}
    for kh in re.split(r'^## ', src, flags=re.M):
        m = re.match(r'Reviewer\s*(\d+)', kh)
        if m:
            neo[m.group(1)] = len(re.findall(r'^\s*(\d+)\.\s+\S', kh, re.M))
    dem = {}
    for r in rows:
        dem[r["reviewer"]] = dem.get(r["reviewer"], 0) + 1
    lech = {k: (neo.get(k), dem.get(k)) for k in set(neo) | set(dem) if neo.get(k) != dem.get(k)}
    kiem(not lech, "Q2a so diem trong so khop THU QUYET DINH goc",
         str(lech) if lech else "%d diem" % len(rows))
    t_thu = chu(os.path.join(GOI, "response-to-reviewers-R1.pdf"))
    kiem(t_thu.count("Action.") == len(rows) and t_thu.count("Where.") == len(rows),
         "Q2b thu co dung mot Action + Where moi diem",
         "%d Action / %d Where / %d diem" % (t_thu.count("Action."),
                                             t_thu.count("Where."), len(rows)))

    def chuan(x):
        """Chi giu chu va so: BO HET gach noi va khoang trang.

        ⛔ Vap 03/10, hai lan. Ngat dong co gach noi tao hai ca KHONG phan
        biet duoc tu van ban: "ORACLE-/GENERATED" (gach noi CO THAT, phai
        giu) va "adap-/tive" (gach noi do LaTeX chen, phai bo). Moi cach noi
        lai deu hong mot trong hai ca. Bo het gach noi o CA HAI VE thi ca hai
        cung khop.
        """
        return re.sub(r'[^a-z0-9]+', '', x.lower())

    t_thu_c = chuan(t_thu)
    thieu_q = [r["ma"] for r in rows
               if chuan(" ".join(r["trich_nguyen_van"].split()[:20]))
               not in t_thu_c]
    kiem(not thieu_q, "Q2c moi trich dan cua so co mat trong thu",
         ", ".join(thieu_q[:4]) if thieu_q else "")

    # ---- Q3 ban to sang co thuc su to sang ----
    t_sach = chu(os.path.join(GOI, "manuscript-R1.pdf"))
    t_tos = chu(os.path.join(GOI, "manuscript-R1-highlighted.pdf"))
    kiem(len(t_tos) > len(t_sach) * 1.01,
         "Q3 ban to sang KHAC ban sach (co phan da xoa)",
         "sach %d ky tu, to sang %d" % (len(t_sach), len(t_tos)))

    # ---- Q4 ban thao va thu dung cung con so ----
    # ⛔ Mot chieu: so nao THU neu thi BAN THAO phai co. Ban dau toi doi ca hai
    # noi deu phai co, va no bao hong vi thu khong trich lai "158/158" du ban
    # thao co. Thu khong can nhac lai moi con so; cai khong duoc phep la thu
    # neu mot con so ma ban thao khong co.
    cap = ["0.010", "0.018", "0.083", "6320", "1.74", "7.99", "0.0038", "0.0105"]
    # Bang trong bai theo kieu IEEE: bo so 0 dau (".083"), con thu viet "0.083".
    # Khac cach in, khong phai mau thuan, nen chap nhan ca hai dang.
    def co(s, t):
        return s in t or (s.startswith("0.") and s[1:] in t)
    sai = [s for s in cap if s in t_thu and not co(s, t_sach)]
    kiem(not sai, "Q4 moi con so THU neu deu co trong BAN THAO",
         "thu neu nhung ban thao khong co: " + ", ".join(sai) if sai else
         "%d/%d con so doi chieu duoc" % (sum(1 for s in cap if s in t_thu), len(cap)))


    # ---- Q5 FRAME ----
    mt = io.open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    m = re.search(r'\\title\{(.*?)\}\s*\n', mt, re.S)
    tde = " ".join(m.group(1).replace("\\\\", " ").split()) if m else ""
    GOC = "KAN-Based Adaptive Parameter Control for Multi-User Satellite FSO/QKD Systems"
    kiem(tde == GOC, "Q5a tieu de KHONG doi", tde[:52])
    intro = io.open(os.path.join(PAPER, "sections", "intro.tex"), encoding="utf-8").read()
    dg = re.findall(r'\\item \\textbf\{([^}]*)\}', intro)
    kiem(len(dg) == 5, "Q5b van du 5 dong gop", "%d dong gop" % len(dg))

    # ---- Q6 nhan float trong thu co that ----
    aux = io.open(os.path.join(PAPER, "main.aux"), encoding="utf-8", errors="replace").read()
    so_tab = {m.group(1) for m in re.finditer(r'\\newlabel\{tab:[^}]+\}\{\{([^}]*)\}', aux)}
    so_fig = {m.group(1) for m in re.finditer(r'\\newlabel\{fig:[^}]+\}\{\{(\d+)\}', aux)}
    dan_t = set(re.findall(r'Table[~\s]+([IVX]+)\b', t_thu))
    dan_f = set(re.findall(r'Fig\.~?(\d+)', t_thu))
    xau = sorted((dan_t - so_tab) | (dan_f - so_fig))
    kiem(not xau and (dan_t or dan_f),
         "Q6 moi so bang/hinh dan trong thu deu TON TAI",
         "khong co: " + ", ".join(xau) if xau else
         "%d bang, %d hinh duoc dan" % (len(dan_t), len(dan_f)))

    # ---- Q7 goi tu dung lai duoc ----
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(os.path.join(GOI, "goi-nop-R1.zip")) as z:
            z.extractall(tmp)
        pkg = os.path.join(tmp, "pkg")
        for f in os.listdir(pkg):
            if f.endswith(".pdf"):
                os.remove(os.path.join(pkg, f))
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
                           cwd=pkg, capture_output=True, text=True)
        out = os.path.join(pkg, "main.pdf")
        n = 0
        if os.path.exists(out):
            info = subprocess.run(["pdfinfo", out], capture_output=True, text=True).stdout
            mm = re.search(r'Pages:\s+(\d+)', info)
            n = int(mm.group(1)) if mm else 0
        log = os.path.join(pkg, "main.log")
        nloi = 0
        if os.path.exists(log):
            nloi = sum(1 for l in io.open(log, encoding="utf-8", errors="replace")
                       if l.startswith("!"))
        kiem(n > 0 and nloi == 0, "Q7 goi TU DUNG LAI duoc tu ban giai nen",
             "%d trang, %d loi" % (n, nloi))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- Q8 kho git ----
    def g(*c):
        return subprocess.run(["git"] + list(c), cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    ban = [l for l in g("status", "--porcelain").splitlines() if l.strip()]
    kiem(not ban, "Q8a kho sach", "%d muc chua commit" % len(ban) if ban else "")
    g("fetch", "-q", "origin")
    kiem(g("rev-parse", "HEAD") == g("rev-parse", "origin/main") != "",
         "Q8b HEAD == origin", g("rev-parse", "--short", "HEAD"))

    print("\n=> %s (%d loi)" % ("DAT" if not loi else "CHUA DAT", len(loi)))
    for x in loi:
        print("   loi: %s" % x)
    return 0 if not loi else 1


def tu_kiem():
    """Doi chung DUONG: tiem loi, cong phai bat."""
    print("== TU KIEM cong QA (doi chung duong) ==\n")
    ok = True
    # 1. them mot tep ghi chu noi bo vao goi
    bia = os.path.join(GOI, "ghi-chu-noi-bo.md")
    io.open(bia, "w").write("ghi chu rieng\n")
    r = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    bat = "HONG Q1b" in r.stdout
    print("  %-44s %s" % ("tiem tep ghi chu noi bo vao goi",
                          "DAT (bat duoc)" if bat else "HONG (cong mu)"))
    ok = ok and bat
    os.remove(bia)
    # 2. sach thi phai im
    r = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    sach = "HONG Q1b" not in r.stdout
    print("  %-44s %s" % ("go ra thi cong im lai", "DAT" if sach else "HONG"))
    ok = ok and sach
    print("\n  => %s" % ("TAT CA DAT" if ok else "CO CA HONG"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
