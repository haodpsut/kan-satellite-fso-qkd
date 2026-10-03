#!/usr/bin/env python3
"""Sinh thu tra loi TCOM-TPS-26-1667 TU SO NHAN XET, de khong the sot diem nao.

⛔ VI SAO SINH THAY VI GO. Bai IoT-J vong truoc tra loi 7/46 diem va an Major lan hai
dung vi chuyen do. O day moi muc trong thu sinh tu MOT DONG cua reviews/so-nhan-xet.csv,
nen so muc trong thu = so dong trong so, kiem duoc bang mot phep dem.

⛔ THU TUC RIENG CUA TCOM (khac IoT-J): editor yeu cau nop thu tra loi dang
**supplementary file**, KHONG phai cover letter, vi cover letter chi editor doc duoc
con phan bien thi khong. Nop nham cho = phan bien khong thay thu.

Hai bai hoc da mang sang tu IoT-J:
  1. **Khong go cung so float.** Dan bang {{nhan}}, so lay tu main.aux luc sinh. Ca that
     30/09: bo hai hinh -> so hinh dich mot bac -> thu van ghi "Fig. 6" trong khi y no
     la hinh nay da thanh Fig. 5, va cong cu khong bat vi no chi hoi "so 6 co ton tai
     khong" chu khong hoi "so 6 con tro dung thu cu khong".
  2. **Cong cam go cung** `Fig./Table/Algorithm + so` trong phan cau tra loi.

⛔ Bo sinh TU CHOI chay neu con diem CHUA CO cau tra loi. Mot thu thieu diem thi tha
khong sinh ra con hon sinh ra roi quen.

    python3 code/gen_response_tcom.py            # sinh thu
    python3 code/gen_response_tcom.py --tien-do  # xem con bao nhieu diem chua viet
"""
import argparse
import csv
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # .../satellite-qkd-kan ; script nam trong code/
SO = os.path.join(ROOT, "reviews", "so-nhan-xet.csv")
AUX = os.path.join(HERE, "paper", "main.aux")
OUT = os.path.join(HERE, "paper", "response-to-reviewers-R1.tex")

TIEU_DE = ("KAN-Based Adaptive Parameter Control for Multi-User Satellite "
           "FSO/QKD Systems")

# ---------------------------------------------------------------------------
# TL[ma] = (hanh dong da lam, vi tri trong ban thao)
# Dan float bang {{nhan}}; KHONG go so.
# ---------------------------------------------------------------------------
TL = {}

TL["R4-2"] = (
    "We agree, and we have narrowed the claim rather than defended it. The phrase "
    "\\emph{information-theoretic security} no longer appears as a property we "
    "establish. Section~{{sec:intro}} now states the scope precisely: the secret-key rate used "
    "throughout is the asymptotic Csisz\\'ar--K\\\"orner fraction "
    "$\\max\\{0, H_2(P_e)-H_2(\\mathrm{QBER})\\}$ per sifted bit, evaluated against the "
    "two attack models we model; it carries no finite-key correction, no composable-"
    "security statement, and no claim against general coherent attacks. The guarantee "
    "reported in the prior work is attributed to that work, under its own assumptions.",
    "Sec.~{{sec:intro}}, paragraph 2")

TL["R4-6"] = (
    "Added. Table~{{tab:notation}} collects every symbol used in "
    "Sections~{{sec:system}}--{{sec:decomposition}}, grouped into decision variables, "
    "geometry and channel, detection and key rate, and solver and controller. Each "
    "entry was checked against the body text, so the table cannot list a symbol the "
    "paper does not use.",
    "Table~{{tab:notation}}, start of Sec.~{{sec:system}}")

TL["R3-5"] = (
    "All three fixed. (i) Reference [9] showed \\texttt{------} because the "
    "bibliography style abbreviates a repeated author list; we disabled that "
    "behaviour, and the entry now prints the full author list. (ii) The abstract no "
    "longer uses the $\\times$ symbol: the gains are written as \\emph{times}. "
    "(iii) We made a consistency pass over the manuscript and corrected a duplicated "
    "word in Section~{{sec:intro}}, two abbreviations that were re-expanded after their first "
    "definition, and an inconsistent rendering of the name Kolmogorov--Arnold.",
    "References; Abstract; Sec.~{{sec:intro}}; Sec.~{{sec:system}}")

TL["R3-3"] = (
    "Redrawn. The single figure-level legend sat above the panel titles and described "
    "the bars of the lower row while being placed over the upper row, whose curves use "
    "different colours and were not in the legend at all. The upper row now carries its "
    "own legend inside the panel, the bar legend sits directly beneath the lower row, "
    "and font sizes were raised. We also corrected something the review did not raise: "
    "the inset box reported a step count next to a label that named the key-rate gain, "
    "mixing two different quantities; the inset now names each explicitly.",
    "Fig.~{{fig:cluster_compare}} and its caption")


TL["R4-1"] = (
    "Both reviews are now cited at the end of the first paragraph, and we say what each "
    "contributes rather than appending them to a list. We note that "
    "\\cite{xu2020rmp} surveys how device imperfections, not protocol design, set the "
    "security of practical QKD, which is precisely the regime our parameter-control "
    "problem lives in; and that \\cite{lu2022micius} reviews the Micius programme, still "
    "the only platform on which satellite QKD has been demonstrated end to end, and the "
    "source of the link budgets that make a multi-user relay architecture worth "
    "optimizing. We verified both entries against Crossref.",
    "Sec.~{{sec:intro}}, end of first paragraph")

TL["R4-5"] = (
    "We have added and discussed the suggested work, but we have not added it as a "
    "numerical baseline, and we would rather give the reason than appear to dodge. "
    "Section~{{sec:intro}} now contrasts it on three axes. Its physical layer is "
    "photon-counting discrete-variable QKD with decoy intensities as the tuned "
    "parameters; ours is intensity-modulated direct detection with a modulation depth "
    "and two detection thresholds, so the parameter spaces do not overlap and a "
    "head-to-head number would compare protocols rather than controllers. It is "
    "single-link, so the coupling that makes our problem hard, one global modulation "
    "depth shared across the cluster, does not arise. And it learns online by "
    "reinforcement whereas we learn offline by supervision from a decomposed oracle. "
    "A reinforcement-learning controller for the multi-user FSO setting is a comparison "
    "worth making, and we state it as future work: constructing a fair one is a study "
    "in itself, and a weak re-implementation would be worse evidence than none.",
    "Sec.~{{sec:intro}}, paragraph before the contributions; Sec.~{{sec:conclusion}}")

TL["R2-1"] = (
    "Justified by measurement, and the question exposed an error we are glad to "
    "correct. While preparing the justification we found that the manuscript stated "
    "$\\tau{=}0.1$ while every run actually used $\\tau{=}0.05$; no result in the paper "
    "was produced at $0.1$. We have corrected the text to the value that was executed. "
    "We then swept $\\tau$ over the oracle solutions of all $250$ clusters: the feasible-"
    "pair count and the aggregate key rate are \\emph{identical} for every "
    "$\\tau \\ge 0.05$, and the constraint only becomes active below "
    "$\\tau \\approx 0.04$, where the feasible count falls from $766$ to $608$. The cap "
    "is therefore an inactive guard in the studied regime rather than a tuned "
    "parameter, and we now say so instead of asserting a value. The correction changes "
    "no number elsewhere in the paper, because $0.05$ is what produced them.",
    "Sec.~{{sec:formulation}}, inter-user secrecy constraint")

TL["R2-2"] = (
    "You identified a real defect, and chasing it down changed the paper. The "
    "standard deviations were of the same magnitude as the values because two "
    "software faults, not two modelling choices, were corrupting the closed-loop "
    "evaluation. "
    "\\textbf{First}, the feature standardiser divided by a per-feature standard "
    "deviation guarded only against exact zero. One controller input is constant in "
    "this regime, with a standard deviation of $7.99\\times10^{-15}$ rather than zero, "
    "so the guard was bypassed and floating-point noise was amplified by a factor of "
    "$1.25\\times10^{14}$; predicted thresholds then moved by $0.35$ in response to "
    "last-bit input differences. "
    "\\textbf{Second}, three of the four controller targets are constant in the "
    "studied regime, so the residual the network was asked to fit had a standard "
    "deviation of $1.74\\times10^{-16}$; the quasi-Newton optimiser diverged on "
    "$20/20$ seeds, producing $6320$ non-finite cells that a silent fallback replaced "
    "with the training median. A diverged model was therefore being reported as a "
    "measurement. "
    "Both are fixed: the standardiser now uses a relative tolerance, and constant "
    "targets are emitted exactly rather than learned. Divergent cells fall from "
    "$6320$ to $0$, the seed standard deviation of the per-user accuracy falls from "
    "$0.034$ to $0.001$, and the reported counts become reproducible across machines. "
    "We also report the divergence count alongside every row, so a diverged run can "
    "never again be read as a measurement.",
    "Sec.~{{sec:results}}, Table~{{tab:cluster}}; artifact "
    "\\texttt{scripts/train\\_cluster.py}")

TL["R1-4"] = (
    "Expanded from five seeds to twenty, and more importantly we now report what the "
    "spread means. The earlier spread was dominated by the two software faults "
    "described in our reply to Reviewer~2, not by model variability: once they are "
    "fixed, the seed standard deviation of the per-user accuracy drops from $0.034$ to "
    "$0.001$ and the closed-loop retention becomes stable to three decimals. We report "
    "mean and standard deviation over twenty seeds, and we additionally report the "
    "number of divergent cells per configuration, because a zero there is what makes "
    "the mean meaningful.",
    "Sec.~{{sec:results}}, Table~{{tab:cluster}}")

TL["R4-7"] = (
    "Agreed, and we have rerun everything at twenty seeds. We also want to be explicit "
    "about something the extra seeds revealed rather than hid: at five seeds the "
    "closed-loop numbers were not reproducible across machines at all, because a "
    "standardisation fault made them sensitive to the fifteenth decimal digit of a "
    "prediction. We traced that causally, by perturbing predictions by a controlled "
    "relative amount and watching the feasible count move, and we fixed the cause "
    "rather than averaging over it. The twenty-seed table is therefore not simply a "
    "larger sample of the old experiment; it is the first run whose numbers are "
    "machine-independent.",
    "Sec.~{{sec:results}}, Table~{{tab:cluster}}")

TL["R4-8"] = (
    "The sample size is unchanged at $250$ clusters, and we now state its consequences "
    "rather than leaving them implicit. With the two faults fixed, the across-seed "
    "standard deviations are $0.001$ for the per-user accuracy and $\\le 0.004$ for "
    "closed-loop retention, so sample size is no longer the limiting source of "
    "uncertainty for those quantities. What $250$ clusters does limit is coverage of "
    "the geometry space, and we say so in the limitations: the clusters are drawn from "
    "a single sampling regime, and we make no claim about behaviour outside it.",
    "Sec.~{{sec:results}}; Sec.~{{sec:conclusion}}, limitations")

TL["R1-1"] = (
    "This is the question the revision is built around, and we can now answer the "
    "second half of it (\\emph{under what conditions does the advantage become "
    "significant}) with a measurement rather than an argument. "
    "The controller has four outputs. Three of them are \\emph{constant} over the "
    "operating regime, because the optimal modulation depth saturates at the physical "
    "ceiling of the intensity-modulation index: sweeping it on a $501$-point grid up "
    "to $0.999$, the optimum sits at the upper boundary in $80\\%$ of sampled channel "
    "conditions, and widening the physical sampling range makes that worse, not "
    "better. On a constant target no model can outperform any other, and a "
    "nine-parameter linear map is exactly as accurate as a network. "
    "The advantage of a learned model is therefore confined to the one output that "
    "genuinely varies. There the numbers are $0.010 \\pm 0.001$ for KAN, "
    "$0.018 \\pm 0.022$ for the regularised MLP and $0.083$ for the linear baseline, "
    "with $589$ parameters against $1386$; but the paired test we have since run over "
    "the shared seeds does \\emph{not} establish the KAN-versus-MLP gap ($p=0.29$), so "
    "we now claim the accuracy advantage only against the linear map, where it is "
    "$20$ of $20$ paired seeds at $p=1.9\\times10^{-6}$. "
    "We have added this characterisation to the results and stated the saturation "
    "explicitly in the limitations, because a reader deciding whether to deploy a "
    "learned controller needs to know which axis is worth learning.",
    "Sec.~{{sec:results}}, Table~{{tab:cluster}}; Sec.~{{sec:conclusion}}")

TL["R2-3"] = (
    "Described, and the honest description is not the one we would have written "
    "before the revision. With the two software faults fixed, KAN and the MLP are "
    "\\emph{indistinguishable} on the closed-loop metric: both reach the oracle rate "
    "on every test configuration, as does the linear baseline, so that metric has "
    "saturated and can no longer separate the models. We say so rather than quoting "
    "it as a win. "
    "Nor, we now find, does accuracy separate them. An exact paired Wilcoxon over the "
    "$20$ shared seeds gives $p=0.29$: on $18$ of $20$ seeds the two are "
    "indistinguishable ($0.0102$ against $0.0103$), and the whole difference in means "
    "comes from $2$ seeds on which the MLP's residual head contributes nothing and it "
    "returns the linear baseline's $0.083$. The KAN does this on no seed, but $2$ of "
    "$20$ against $0$ of $20$ is itself not significant (Fisher $p=0.49$), so we report "
    "it as an observation. What separates the classes is parameter count, $43\\%$ of "
    "the MLP's, and symbolic extractability. "
    "Our reading: the MLP is the better choice when the target has broad support and "
    "retraining is cheap; the KAN is the better choice here, where the learnable "
    "signal is confined to one output, the parameter budget is on-board, and a single "
    "bad initialisation cannot be detected in flight.",
    "Sec.~{{sec:results}}, Table~{{tab:cluster}}")

TL["R3-4"] = (
    "Justified, and the revision strengthened the case for keeping the linear "
    "baseline rather than weakening it. Three of the four controller outputs are "
    "constant in this regime, and on those a nine-parameter linear map attains the "
    "same accuracy as either network; without that baseline in the table a reader "
    "could not see that, and would credit the networks for accuracy that requires no "
    "network at all. The linear model also reaches the oracle rate on every test "
    "configuration, so it is not a straw man. We kept the regularised MLP as the "
    "like-for-like learned competitor and report its parameter count alongside, since "
    "parameter efficiency is one of the axes under comparison. We did not add further "
    "learned baselines: tuning a third architecture on our scenario would make the "
    "comparison depend on how well we tuned someone else's model, and we would rather "
    "state that limit than hide it behind an under-tuned row.",
    "Sec.~{{sec:results}}, Table~{{tab:cluster}}")

TL["R3-1"] = (
    "Two separate answers, and the first one is no. "
    "\\textbf{On comprehensiveness:} URA and BSA are \\emph{not} a comprehensive "
    "treatment of Eve, and the manuscript should not have left that implicit. They are "
    "the two attacks for which this physical layer admits a closed-form detectability "
    "criterion, and they are modelled \\emph{individually}, not jointly. Not covered: "
    "photon-number-splitting on the weak-coherent source, Trojan-horse probing of the "
    "modulator, detector blinding or time-shift attacks on the dual-threshold receiver, "
    "side channels in classical post-processing, and any collective or coherent attack. "
    "We now say this explicitly rather than by omission, and list those classes as "
    "future work. Section~{{sec:intro}} also narrows the security language accordingly, "
    "as Reviewer~4 asked. "
    "\\textbf{On the BSA detail:} we have added what the criterion actually depends on. "
    "The splitter is passive and draws a fixed power fraction "
    "$\\eta_{\\mathrm{leak}}$ at the relay; detectability enters only through the loss it "
    "imposes on the legitimate sifting probability, "
    "$m_{\\mathrm{BSA}}(\\beta)=1-\\Psift(\\gamma(1-\\eta_{\\mathrm{leak}}))/\\Psift(\\gamma)$, "
    "so Eve's own receiver is \\emph{not} modelled, only the trace she leaves. "
    "We also swept the splitter ratio, which the manuscript had fixed without "
    "justification: the fraction of pairs for which detection can be certified falls "
    "from $100\\%$ at $\\eta_{\\mathrm{leak}}\\ge1.4\\%$ to $73\\%$ at $1.25\\%$ and to "
    "zero at $1.15\\%$, so the whole transition occupies about $0.25$ percentage points. "
    "The design therefore certifies detection of a splitter drawing $1.4\\%$ or more; "
    "against a stealthier one the constraint is not met and no key is emitted. That is "
    "the conservative behaviour, but it is also a real limit, and it is now stated.",
    "Sec.~{{sec:formulation}}, URA and BSA subsections; Sec.~{{sec:conclusion}}")

TL["R1-2"] = (
    "Strengthened where a proof exists, and sharpened where one does not. The two "
    "halves of the solver had been carrying the same word, \\emph{exact}, while having "
    "very different status, and we have separated them. "
    "Section~{{sec:decomposition}} now gives the one-line argument that the outer/inner "
    "split is lossless: with $\\boldsymbol{\\beta}$ ranging over a compact box and the "
    "objective continuous on it, partial maximization gives "
    "$\\max \\Rf = \\max_{(\\mu,\\chi)}[\\max_{\\boldsymbol{\\beta}} \\Rf]$ exactly. No "
    "relaxation and no assumption beyond compactness is involved, so this part needs no "
    "empirical support at all. "
    "What that identity does not give is that our inner routine attains the inner "
    "maximum. We now name the exact step that lacks a guarantee: the per-user "
    "one-dimensional maximizations are exhaustive on a grid plus refinement and so "
    "attain their own optimum, while the \\emph{mean-field coupling between users} has "
    "no contraction proof. For that step we report evidence and label it as evidence: "
    "agreement with a brute-force joint search at $N{=}2$, convergence within six "
    "iterations, and the initialization study added for Reviewer~4. We would rather "
    "mark that boundary than let one adjective cover both sides of it.",
    "Sec.~{{sec:decomposition}}")

TL["R1-5"] = (
    "The limitations subsection has been rewritten rather than extended, because one of "
    "its existing entries turned out to be wrong in a way worth flagging. It attributed "
    "the cluster controller's seed variance to the oracle sitting on a sharp "
    "four-dimensional feasibility boundary. That was a scientific explanation for what "
    "we have since traced to two software faults, and we have replaced it with the "
    "mechanism and the measurements. "
    "On oracle dependence specifically, which you asked about: the controller is "
    "trained by supervision from the decomposed solver, inherits whatever that solver "
    "gets wrong, and cannot by construction exceed it. We now state this together with "
    "a consequence we would otherwise have been tempted to present as a result: once "
    "the faults are fixed, every controller we tested, including a nine-parameter "
    "linear map, reaches the oracle rate on every test configuration, so the "
    "closed-loop retention metric has saturated and no longer separates architectures. "
    "Three further limitations are now stated explicitly: the degeneracy of three of "
    "the four targets and what it costs; the scope of the eavesdropper model, which "
    "covers two attacks modelled individually; and the sharpness of the "
    "beam-splitting detection threshold, which spans about a quarter of a percentage "
    "point.",
    "Sec.~{{sec:conclusion}}, limitations")

TL["R4-3"] = (
    "You were right that the convergence evidence was weak, and testing it properly "
    "overturned one of our statements. We report what we can prove and what we "
    "measured, separately. "
    "\\textbf{Proved:} the outer/inner split is a lossless partial maximization, which "
    "needs only compactness of the threshold box and continuity of the objective; the "
    "argument is now given in Section~{{sec:decomposition}}. The per-user "
    "one-dimensional subproblems are solved by exhaustive grid plus refinement and so "
    "attain their own optimum. "
    "\\textbf{Measured, and it contradicts our earlier claim:} we had written that the "
    "mean-field fixed point was unique on every cluster tested. Restarting the inner "
    "iteration from one common-threshold value at a time, \\emph{all 12 of 12} tested "
    "cluster states reach \\emph{different} fixed points depending on the start, with "
    "the Alice-side threshold differing by up to $1.31$. The uniqueness we observed was "
    "an artefact of the solver already running four starts and keeping the best. We "
    "have retracted the uniqueness statement, and we now say that multi-start is a "
    "requirement rather than a convenience. "
    "\\textbf{Not provided:} a contraction proof or local convergence rate for the "
    "mean-field coupling. We would rather state that gap and give the counterexample "
    "count than offer a monotonicity argument we cannot establish for the coupled map.",
    "Sec.~{{sec:decomposition}}; artifact \\texttt{scripts/do\\_trien\\_khai.py}")

TL["R4-4"] = (
    "Added, and the measurement argues against one motivation we might have claimed. "
    "On eight independent passes we recomputed the optimum at every step and then "
    "asked what it costs to hold parameters for longer. Refreshing every $50$~s is the "
    "reference; holding for $100$~s retains $88.9 \\pm 3.5\\%$ of the key, $150$~s "
    "retains $84.4 \\pm 2.0\\%$, $250$~s retains $68.3 \\pm 9.4\\%$, and beyond about "
    "$600$~s the link is mostly lost. Re-optimization every one to two steps is "
    "therefore the operating requirement. "
    "The oracle itself costs $7.57 \\pm 2.15$~s per step on the reported server, which "
    "is well inside that budget. We therefore do \\emph{not} claim the learned "
    "controller is needed for latency on this hardware, and we have removed any "
    "suggestion of it. Its case rests on operating without the full physics model and "
    "channel state at the node, and on a bounded, deterministic inference cost. "
    "One further observation we did not expect: the retention curve is not monotone in "
    "the refresh interval, because what matters is where the refresh instants fall "
    "relative to the geometry of the pass, not how many of them there are. A "
    "fixed-clock refresh policy is therefore the wrong design; refresh should be "
    "triggered by change in the channel state. We say so rather than quoting a single "
    "cadence number.",
    "Sec.~{{sec:results}}; Sec.~{{sec:conclusion}}")

TL["R3-2"] = (
    "Expanded, and we tried to make the addition do work rather than lengthen a list. "
    "The survey now covers the modulation-signalling branch specifically and says, for "
    "each entry, what it settles and what it leaves open for us. Subcarrier-wave QKD "
    "has been analysed numerically for free-space ship-to-port links "
    "\\cite{goncharov2025scw} and routed through a deployed metropolitan transport "
    "network \\cite{tarabrina2023scw}, which puts the signalling format outside the "
    "laboratory but in both cases on a \\emph{single} link rather than a shared relay. "
    "Direct intensity and phase modulation has been demonstrated as a QKD transmitter "
    "\\cite{liu2026intensity}, which is the hardware assumption our modulation depth "
    "parameterizes. The free-space continuous-variable line is surveyed in "
    "\\cite{kargina2026cvqkd}; it shares our reliance on classical coherent components "
    "but replaces the dual-threshold decision with homodyne detection, so the parameter "
    "that dominates our problem has no counterpart there. We also added two review "
    "articles at Reviewer~4's suggestion and a recent learning-based controller at "
    "their request. Every entry was verified against Crossref before being cited.",
    "Sec.~{{sec:intro}}, paragraphs on the modulation-signalling branch")

TL["R1-3"] = (
    "Added as a new subsection, covering all four axes you named. We re-solved the "
    "oracle on $25$ fresh clusters under each of sixteen conditions: turbulence (wind "
    "$11$ to $41$~m/s in the Hufnagel--Valley profile), serving geometry (zenith bands "
    "from $0$--$4^\\circ$ to $20$--$35^\\circ$), cluster size ($N{=}2$ to $6$), and "
    "eavesdropper standoff ($10$ to $120$~m). "
    "Two things survive, and the second was not what we expected. "
    "\\textbf{First}, the modulation-depth saturation is a property of the problem "
    "rather than of one regime: the optimum sits at its upper bound in $100\\%$ of "
    "clusters in every feasible condition, across every turbulence level, zenith band "
    "and cluster size. The one exception is a close eavesdropper, where at a $10$--$20$~m "
    "standoff the URA constraint binds and only $52\\%$ remain at the bound. "
    "\\textbf{Second}, turbulence \\emph{helps} security here, which we report as a "
    "design constraint and not as a result in our favour. Two conditions yielded no "
    "feasible cluster: the $20$--$35^\\circ$ zenith band, for the expected reason of a "
    "longer slant path, and the \\emph{low}-turbulence case at $11$~m/s. The reason is "
    "that a beam splitter is detected only through the loss it imposes on the sifting "
    "statistics, so detectability scales with the width of the fading distribution. At "
    "a fixed geometry $m_{\\mathrm{BSA}}$ rises monotonically with turbulence: $0.0038$ "
    "at $11$~m/s, which is below the $0.005$ threshold, then $0.0063$, $0.0089$ and "
    "$0.0105$ at $21$, $31$ and $41$~m/s. A clear, still night is therefore the worst "
    "case for this criterion, not the best.",
    "Sec.~{{sec:results}}, robustness subsection")

# ---------------------------------------------------------------------------
def doc_aux():
    if not os.path.exists(AUX):
        raise SystemExit("⛔ thieu %s: dung ban thao truoc khi sinh thu." % AUX)
    a = io.open(AUX, encoding="utf-8", errors="replace").read()
    return {m.group(1): m.group(2) for m in
            re.finditer(r"\\newlabel\{((?:fig|tab|alg|sec|eq):[^}]+)\}\{\{([^}]*)\}", a)}


def thay_nhan(van, so_nhan):
    thieu = []

    def f(m):
        k = m.group(1)
        if k not in so_nhan:
            thieu.append(k)
            return "??"
        return so_nhan[k]
    ra = re.sub(r"\{\{([^}]+)\}\}", f, van)
    if thieu:
        raise SystemExit("⛔ nhan khong co trong main.aux: %s"
                         % ", ".join(sorted(set(thieu))))
    return ra


def esc(t):
    return t


def nap_so():
    with io.open(SO, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tien_do(rows):
    xong = [r["ma"] for r in rows if r["ma"] in TL]
    chua = [r["ma"] for r in rows if r["ma"] not in TL]
    print("== Tien do thu tra loi TCOM-TPS-26-1667 ==\n")
    print("  da viet %d/%d diem\n" % (len(xong), len(rows)))
    theo_goi = {}
    for r in rows:
        theo_goi.setdefault(r["goi"], []).append((r["ma"], r["ma"] in TL))
    for goi in sorted(theo_goi):
        ds = theo_goi[goi]
        n = sum(1 for _, ok in ds if ok)
        print("  goi %s: %d/%d  %s" % (goi, n, len(ds),
              " ".join(("✅" if ok else "⬜") + m for m, ok in ds)))
    if chua:
        print("\n  CHUA viet: %s" % ", ".join(chua))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tien-do", action="store_true")
    a = ap.parse_args()
    rows = nap_so()
    if a.tien_do:
        return tien_do(rows)

    # ---- cong 1: moi dong trong so phai co cau tra loi, va nguoc lai ----
    thieu = [r["ma"] for r in rows if r["ma"] not in TL]
    thua = [k for k in TL if k not in {r["ma"] for r in rows}]
    if thieu or thua:
        print("⛔ SO va THU khong khop, KHONG sinh thu.")
        if thieu:
            print("   chua co cau tra loi cho %d diem: %s"
                  % (len(thieu), ", ".join(thieu)))
        if thua:
            print("   cau tra loi thua, khong co trong so: %s" % ", ".join(thua))
        print("\n   Chay `--tien-do` de xem con thieu nhung goi nao.")
        return 1

    # ---- cong 2: cam go cung so float ----
    go_cung = []
    for ma, (h, v) in sorted(TL.items()):
        for van in (h, v):
            for m in re.finditer(
                    r"(?:Fig\.|Figure|Table|Alg\.|Algorithm|Sec\.|Section)~?\s*"
                    r"([0-9]+|[IVX]+)\b", van):
                go_cung.append("%s: %s" % (ma, m.group(0)))
    if go_cung:
        print("⛔ TL con GO CUNG so float/muc, phai dan bang {{nhan}}:")
        for x in go_cung[:10]:
            print("   %s" % x)
        return 1

    so_nhan = doc_aux()
    n_r = {}
    for r in rows:
        n_r[r["reviewer"]] = n_r.get(r["reviewer"], 0) + 1

    L = [r"% Sinh boi code/gen_response_tcom.py tu reviews/so-nhan-xet.csv. KHONG sua tay.",
         r"% ⛔ NOP DANG SUPPLEMENTARY FILE, khong phai cover letter (yeu cau cua editor).",
         r"\documentclass[10pt]{article}",
         r"\usepackage[margin=1in]{geometry}",
         r"\usepackage{amsmath,amssymb}",
         r"\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue]{hyperref}",
         r"\usepackage{enumitem}", r"\setlist{nosep}",
         r"% macro dung chung voi main.tex: thieu chung thi LaTeX nem loi ma bo sinh",
         r"% van bao thanh cong. Da vap 03/10: 21/21 muc nhung 4 loi Undefined control sequence.",
         r"\newcommand{\Rf}{R_f}", r"\newcommand{\rs}{R_s}",
         r"\newcommand{\Pe}{P_{\mathrm{err}}}", r"\newcommand{\Pc}{P_{\mathrm{corr}}}",
         r"\newcommand{\Psift}{P_{\mathrm{sift}}}", r"\newcommand{\QBER}{\mathrm{QBER}}",
         r"\emergencystretch=1em",
         r"\newcommand{\rev}[1]{\par\medskip\noindent\textit{#1}\par\smallskip}",
         r"\newcommand{\act}[2]{\noindent\textbf{Action.} #1\par\noindent\textbf{Where.} #2\par}",
         r"\title{Response to Reviewers\\ \large TCOM-TPS-26-1667 --- First Revision}",
         r"\author{\normalsize " + TIEU_DE + "}",
         r"\date{}", r"\begin{document}", r"\maketitle", "",
         r"\section*{Before the point-by-point replies}", "",
         r"We thank the Associate Editor and the four reviewers. This letter answers "
         r"\textbf{every} numbered point: %s, %d in total. It is generated from a ledger "
         r"with one row per comment, so the count can be checked rather than trusted."
         % (", ".join("%d from Reviewer~%s" % (n_r[k], k) for k in sorted(n_r)),
            len(rows)), ""]

    for rv in sorted(n_r):
        rs = [r for r in rows if r["reviewer"] == rv]
        L += [r"\subsection*{Reviewer %s --- %d comment%s}"
              % (rv, len(rs), "" if len(rs) == 1 else "s"), ""]
        for i, r in enumerate(rs, 1):
            q = " ".join(r["trich_nguyen_van"].split())
            q = esc(q[:430]) + (r"\dots" if len(q) > 430 else "")
            h, v = TL[r["ma"]]
            L += [r"\paragraph{Comment %d.}" % i, r"\rev{%s}" % q,
                  r"\act{%s}{%s}" % (thay_nhan(h, so_nhan), thay_nhan(v, so_nhan)), ""]

    L += [r"\end{document}", ""]
    io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("✅ ghi %s" % os.path.relpath(OUT, ROOT))
    print("   %d muc, khop tung dong cua so-nhan-xet.csv" % len(rows))

    # ---- cong 3: TU DICH THU ----
    # ⛔ Vap 03/10: bo sinh bao "21/21 muc" trong khi LaTeX nem 4 loi "Undefined
    # control sequence" (macro \Rf, \Psift chi dinh nghia trong main.tex). Mot bo
    # sinh bao thanh cong ma dau ra khong dich duoc thi loi bao ay vo nghia.
    import shutil
    import subprocess
    if shutil.which("latexmk") is None:
        print("   ⚠ khong co latexmk, KHONG tu kiem duoc ban dich")
        return 0
    thu_muc = os.path.dirname(OUT)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                    os.path.basename(OUT)], cwd=thu_muc,
                   capture_output=True, text=True)
    log = os.path.join(thu_muc,
                       os.path.splitext(os.path.basename(OUT))[0] + ".log")
    nd = io.open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    loi = [l for l in nd.splitlines() if l.startswith("!")]
    over = [l for l in nd.splitlines() if "Overfull" in l]
    if loi:
        print("   ⛔ ban dich co %d LOI LaTeX, thu KHONG dung duoc:" % len(loi))
        for l in loi[:5]:
            print("      %s" % l)
        return 1
    print("   ✅ dich sach: 0 loi, %d Overfull" % len(over))
    return 0


if __name__ == "__main__":
    sys.exit(main())
