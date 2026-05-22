# Formulation — Multi-User Satellite FSO/QKD with DT/DD

This document derives, transaction-paper style, every formula implemented in
`src/satqkd/`. It follows the model of the three source papers (the lineage by
Vu, Pham, Dang, Le, Pham), unifying:

- **[P1]** Vu *et al.*, "Design and Performance of Relay-Assisted Satellite FSO/QKD Systems," *IEEE Access*, 2020.
- **[P2]** Vu *et al.*, "Toward Practical Entanglement-Based Satellite FSO/QKD Systems Using DT/DD," *IEEE Access*, 2022.
- **[P3]** Vu *et al.*, "Design of Satellite-Based FSO/QKD Systems Using GEO/LEOs for Multiple Wireless Users," *IEEE Photonics J.*, 2023.

Our target system is the GEO→LEO→multi-user architecture of **[P3]**, with the
pointing-error model of **[P1]** re-introduced (dropped in [P3]), evaluated over a
time-varying LEO pass. This file covers the *static, single-time-instant*
physical layer that forms the optimization "environment". Time variation,
handover and the KAN controller are documented separately.

Notation: $U \in \{A, B_i\}$ (Alice / user $i$); $C$ = Charlie (GEO source);
$E$ = eavesdropper. Bits are equiprobable, $P_C(0)=P_C(1)=\tfrac12$.

---

## 1. Signal model

Charlie SIM/BPSK-modulates a CW laser with intensity modulation depth
$\mu \in (0,1)$ (called $\alpha$ in [P1], $\mu$ here):

$$P_s(t) = \tfrac{P}{2}\,[\,1 + \mu\, m(t)\,], \qquad m(t)=A_c g(t)\cos(2\pi f_c t + \pi d),\ d\in\{0,1\}. \tag{1}$$

The LEO relay applies optical amplify-and-forward gain $G_a$ (EDFA). After the
PIN photodetector (responsivity $R_e$), OBPF and BPSK demodulation, the DC term is
removed and the decision current for transmitted bit $d$ is **[P3, Eq. 3]**:

$$
i_0^U = -\tfrac14 \mu R_e P\, G_a\, h_{e2e}^U + n^U,\qquad
i_1^U = +\tfrac14 \mu R_e P\, G_a\, h_{e2e}^U + n^U. \tag{2}
$$

The modulation depth $\mu$ scales the signal separation: **smaller $\mu$ ⇒ the two
levels collapse toward zero**, which is exactly the lever that raises Eve's error
(Section 6). $h_{e2e}^U$ is the end-to-end channel coefficient (Section 2);
$n^U$ is zero-mean AWGN with variance $(N^U)^2$ (Section 3).

Define the **mean peak signal current** (using $\mathbb{E}[h_a]=1$, Section 2.3):

$$ s^U \triangleq \tfrac14 \mu R_e P\, G_a\, \bar h^U, \qquad \bar h^U = h_g^U h_l^U \tag{3} $$

where $\bar h^U$ is the *deterministic* part of the channel (geometric + atmospheric
attenuation) and the random turbulence $h_a$ (mean 1) multiplies it instantaneously.

---

## 2. Channel model  $h_{e2e}^U = h_g^U\, h_l^U\, h_a^U$

End-to-end channel = (GEO→LEO geometric) × (LEO→user geometric) × (attenuation) ×
(turbulence). GEO→LEO is treated as loss-only at the beam center **[P3, Sec. III-A]**.

### 2.1 Geometric spreading (Gaussian beam) — [P3, Eq. 6–7], [P1, Eq. 7]

Normalized transmit intensity at link distance $L$:

$$ I_{\text{beam}}(\boldsymbol\rho; L) = \frac{2}{\pi w_L^2}\exp\!\Big(-\frac{2\|\boldsymbol\rho\|^2}{w_L^2}\Big). \tag{4}$$

Beam radius at distance $L$ from a transmitter of beam waist $w_0=\lambda/(\pi\theta)$,
divergence $\theta = 2.44\,\lambda/D$ (aperture $D$):

$$ w_L = w_0\Big[\,1 + \big(\tfrac{\lambda L}{\pi w_0^2}\big)^2\,\Big]^{1/2}. \tag{5}$$

The fraction of power collected by a receiver of radius $a$ at radial offset $r$ from
the footprint center is **[P1, Eq. 7]**:

$$ h_g(r) \approx A_0\,\exp\!\Big(-\frac{2 r^2}{w_{L,\text{eq}}^2}\Big),\qquad
A_0 = [\operatorname{erf}(\upsilon)]^2,\quad \upsilon = \frac{\sqrt\pi\, a}{\sqrt2\, w_L}, \tag{6}$$

$$ w_{L,\text{eq}}^2 = w_L^2\,\frac{\sqrt\pi\,\operatorname{erf}(\upsilon)}{2\,\upsilon\,e^{-\upsilon^2}}. \tag{7}$$

For a legitimate user at the beam center $r=0$: $h_g=A_0$. For Eve at offset
$r=d_E$ the **geometric leakage ratio** is

$$ \frac{h_g^E}{h_g^U} = \exp\!\Big(-\frac{2 d_E^2}{w_{L,\text{eq}}^2}\Big). \tag{8}$$

Because the satellite footprint is hundreds of metres wide, this ratio is close to 1
for realistic $d_E$ (tens of metres) — which is *why* security must come from small
$\mu$ rather than from geometry alone.

### 2.2 Atmospheric attenuation — Beer–Lambert [P3, Eq. 8–9]

$$ h_l = \exp(-\sigma\, L_{h\text{-}U}),\quad
\sigma(\lambda)=3.912\Big(\tfrac{\lambda[\text{nm}]}{550}\Big)^{-q(V)}\!/V[\text{km}], \tag{9}$$

$$ q(V)=\begin{cases}1.6 & V>50\,\text{km}\\ 1.3 & 6<V<50\,\text{km}\\ 0.585\,V^{1/3} & V<6\,\text{km}\end{cases}, \qquad
L_{h\text{-}U}=\frac{H_h-H_U}{\cos\theta_U}, \tag{10}$$

with $H_h=20$ km the top of the attenuating layer and $\theta_U$ the zenith angle.

### 2.3 Turbulence-induced fading — log-normal (weak regime) [P3, Eq. 10–11]

For LEO-to-ground with elevation $\ge 30^\circ$ (zenith $\le 60^\circ$) turbulence is
weak, so $h_a$ is log-normal with unit mean:

$$ f_{h_a}(h_a)=\frac{1}{\sqrt{8\pi}\,h_a\,\sigma_X}\exp\!\Big(-\frac{(\ln h_a - 2\mu_X)^2}{8\sigma_X^2}\Big),\quad \mu_X=-\sigma_X^2. \tag{11}$$

(Here $h_a=e^{2X}$, $X\sim\mathcal N(\mu_X,\sigma_X^2)$; $\mu_X=-\sigma_X^2$ enforces
$\mathbb E[h_a]=1$.) The log-amplitude variance is the (quarter-)Rytov variance
**[P3, Eq. 11]**:

$$ \sigma_X^2 = 0.56\,k^{7/6}\sec^{11/6}(\theta_U)\int_{H_U}^{H_C} C_n^2(h)\,(h-H_U)^{5/6}\,dh,\quad k=\tfrac{2\pi}{\lambda}, \tag{12}$$

with the Hufnagel–Valley profile **[P3, Eq. 19]**:

$$ C_n^2(h)=0.00594\big(\tfrac{w}{27}\big)^2(10^{-5}h)^{10}e^{-h/1000}
+2.7\times10^{-16}e^{-h/1500}+C_n^2(0)\,e^{-h/100}. \tag{13}$$

---

## 3. Receiver noise — [P3, Eq. 4]

$$ (N^U)^2 = \underbrace{(\sigma_{sh}^U)^2}_{\text{shot}} + \underbrace{(\sigma_{bL})^2+(\sigma_{bU})^2}_{\text{background}} + \underbrace{(\sigma_{aL})^2}_{\text{ASE}} + \underbrace{(\sigma_{th}^U)^2}_{\text{thermal}}, \tag{14}$$

$$
(\sigma_{sh}^U)^2 = 2qR_e\big(\tfrac14 P G_a \bar h^U\big)\Delta f,\quad
(\sigma_{bU})^2 = 2qR_e P_b^U \Delta f,\quad
(\sigma_{th}^U)^2 = \tfrac{4k_BT F_n}{R_L}\Delta f,
$$
$$
(\sigma_{aL})^2 = 2q P_a^L \bar h^U \Delta f,\quad P_a^L=\tfrac{hc}{\lambda}(n_{sp}-1)G_a B_0,\quad
P_b^U=\Re_\odot a_U^2\,\Im,\quad \Im=\tfrac{B_0\lambda^2}{2c},
$$
with $\Delta f = R_b/2$. (See `params.py` for symbols.)

---

## 4. Decision statistics in terms of a single SNR ratio

The DT thresholds are symmetric about zero **[P3, Eq. 15–16]**:

$$ d_0^U = \mathbb E[i_0^U] - \beta^U N^U = -s^U - \beta^U N^U,\qquad
d_1^U = +s^U + \beta^U N^U, \tag{15}$$

where $\beta^U$ is the **DT scale coefficient** (a key design knob). Define the
**normalized mean SNR amplitude**

$$ \boxed{\ \gamma^U \triangleq \frac{s^U}{N^U} = \frac{\tfrac14\mu R_e P G_a \bar h^U}{N^U}\ } \tag{16}$$

so $\gamma^U \propto \mu$. The conditional probabilities **[P3, Eq. 13–14]**
$P_{U|C}(y|x)=\int Q(\cdot)f_{h_a}\,dh_a$ collapse, after substituting (15) and
$i_x(h_a)=\pm s^U h_a$, to **just two integrals**:

$$ P_{\text{corr}}^U \triangleq P_{U|C}(1|1)=P_{U|C}(0|0)=\mathbb E_{h_a}\!\big[\,Q\big(\beta^U + \gamma^U(1-h_a)\big)\,\big], \tag{17}$$

$$ P_{\text{err}}^U \triangleq P_{U|C}(0|1)=P_{U|C}(1|0)=\mathbb E_{h_a}\!\big[\,Q\big(\beta^U + \gamma^U(1+h_a)\big)\,\big], \tag{18}$$

with $Q(x)=\tfrac12\operatorname{erfc}(x/\sqrt2)$. This $(\gamma,\beta)$ reduction is the
clean interface the optimizer/KAN will act on.

### Gauss–Hermite evaluation of (17)–(18) — [P3, Eq. 30]

With $y=\dfrac{\ln h_a - 2\mu_X}{\sqrt8\,\sigma_X}$ the log-normal expectation becomes a
standard Gauss–Hermite quadrature (physicists' weight $e^{-y^2}$):

$$ \mathbb E_{h_a}[\,g(h_a)\,] \approx \frac{1}{\sqrt\pi}\sum_{j=1}^{n} w_j\, g\big(h_a(x_j)\big),\quad
h_a(x_j)=\exp\!\big(2\sqrt2\,\sigma_X x_j - 2\sigma_X^2\big), \tag{19}$$

converging by $n=20$ nodes **[P2, Eq. 26]**. (Verified: $\mathbb E[h_a]=e^{a^2/4-2\sigma_X^2}=1$ with $a=2\sqrt2\sigma_X$.)

---

## 5. Sift probability, QBER (single Charlie→user link) — [P3, Eq. 12, 25]

$$ P_{\text{sift}}^U = \sum_{x,y}P_C(x)P_{U|C}(y|x)=P_{\text{corr}}^U+P_{\text{err}}^U, \tag{20}$$

$$ \text{QBER}^U = \frac{P_{\text{err}}^U}{P_{\text{sift}}^U}. \tag{21}$$

Design requirements **[P3, Sec. V-C]**: $P_{\text{sift}}>10^{-3}$ (Mb/s sifted key at
Gb/s line rate) **and** $\text{QBER}<10^{-3}$ (correctable). These become *constraints*
in the optimization.

---

## 6. Eve's error probability (URA) — [P2, Eq. 32–35]

Eve cannot know $\beta$, so she uses the optimal threshold $d_E=0$. With her own ratio
$\gamma^E = s^E/N^E$ (where $s^E\propto h_g^E$ via Eq. 8):

$$ P_{\text{err}}^E = \mathbb E_{h_a}\!\big[\,Q(\gamma^E h_a)\,\big]. \tag{22}$$

As $\mu\to0$, $\gamma^E\to0$ and $P_{\text{err}}^E\to Q(0)=\tfrac12$. Security target:
$P_{\text{err}}^E > 0.1$ **[P3, Sec. V-B]** — the third constraint, coupling back to $\mu$.

---

## 7. Multiple users — [P3, Eq. 17–24]

### TDMA baseline
$P_{\text{sift}}^{AB_i}=P_{AB_i}(0,0)+\dots$, with sifted-rate $R_s^i=P_{\text{sift}}^{AB_i}R_b/N$ (split among $N$ users).

### Proposed fading-randomness multiple access [P3, Eq. 19–24]
Charlie sends the *same* sequence to all users; each Alice–$B_i$ pair keeps only its
own (rarely overlapping) sifted bits, excluding mutual bits with ratio $\chi\in[0,1]$:

$$ P_{\text{sift}}^{AB_i\text{-excl}} = P(AB_i) - \chi\,P(AB_i)_{\text{excl}}, \tag{23}$$

$$ P(AB_i)_{\text{excl}}\approx\sum_{k=0}^{N-2}(-1)^k C_{N-1}^{k+1}\big[P_{A|C}(0|0)P_{B_i|C}(0|0)\big]^{\,k+2}, \tag{24}$$

(inclusion–exclusion, proved by induction in [P3, App. B]).

### Final key-creation rate — [P3, Eq. 29]
$$ R_i^f = R_s^i\big[\,I(A;B_i) - \max\big(I(A;E_1),I(B_i;E_2),I(E_1;E_2)\big)\,\big],\quad
R_f=\sum_{i=1}^{N}R_i^f, \tag{25}$$
with mutual informations $I(\cdot;\cdot)$ from the binary-erasure-channel model
**[P3, Eq. 28]**.

---

## 8. The optimization problem (our contribution)

At each time instant $t$ of a LEO pass the geometry $(\theta_U(t), L(t), \sigma_X^2(t))$
is known from TLE. We choose $\mathbf{p}(t)=(\mu, \beta_A, \{\beta_{B_i}\}, \chi)$ to

$$
\max_{\mathbf p(t)}\ R_f(t;\mathbf p)\quad\text{s.t.}\quad
\text{QBER}^U<10^{-3},\ P_{\text{sift}}^U>10^{-3},\ P_{\text{err}}^E>0.1,\ \text{(BSA detectable)}. \tag{26}$$

No closed form (multi-user coupling via $\chi$, Gauss–Hermite integrals,
time-varying geometry) ⇒ we learn a **KAN controller** $\mathbf p^\star=\mathcal K(\theta,\sigma_X^2,L,N)$
plus a fast KAN **surrogate** of $R_f$ for onboard real-time adaptation, and extract
closed-form design rules from the trained KAN splines. The analytical code here is the
ground-truth generator and the Monte-Carlo validator for that controller.
