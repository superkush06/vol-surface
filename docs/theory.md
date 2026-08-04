# The mathematics behind `volsurf`

This is the reasoning the code implements, in the order the code needs it:
what an option surface is a statement about, how to parameterise it, what
makes a parameterisation admissible, and why each algorithm here was chosen
over the obvious alternative. Every formula below appears in the module named
beside it.

Whether the code actually implements it is a separate question, answered in
[`docs/validation.md`](validation.md): every formula here that has a published
closed form, a degenerate limit, or a simulable model behind it is compared
against one there.

---

## 1. A smile is a claim about a distribution

Fix an expiry $T$. Under the risk-neutral measure a call is the discounted
expectation of its payoff, so

$$C(K) = e^{-rT}\,\mathbb{E}\!\left[(S_T - K)^+\right]
       = e^{-rT}\int_K^\infty (s - K)\, q(s)\, \mathrm{d}s .$$

Differentiate twice in $K$ — the Breeden–Litzenberger identity (1978):

$$\frac{\partial^2 C}{\partial K^2} = e^{-rT} q(K).$$

The second strike-derivative of the call curve *is* the risk-neutral density,
up to discounting. That one line drives everything else in this library:

- A smile is not a curve you may draw freely. It is a density in disguise.
- $\partial^2 C/\partial K^2 \ge 0$ is not a modelling preference. A negative
  value is a butterfly spread — long $K-h$, short two $K$, long $K+h$ — with
  non-negative payoff and negative cost.
- The condition is a statement about *prices at one forward*, which is why
  `butterfly_violations` takes `forward` as a required argument. Pricing each
  strike triple at its own middle strike, as if spot moved between quotes,
  answers a different and wrong question.

## 2. Black-Scholes as a quoting convention — `black_scholes.py`

With continuous dividends $q$,

$$C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad
d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt{T}} .$$

Nobody in the options market believes the lognormal model. They use it as a
change of variables: an invertible map from price to a number, $\sigma$, that
is comparable across strikes and maturities. Everything downstream models
$\sigma(K, T)$, not price.

## 3. Inverting the convention — `iv.py`

$C(\sigma)$ is strictly increasing on $(0,\infty)$, since

$$\frac{\partial C}{\partial \sigma} = S e^{-qT}\varphi(d_1)\sqrt{T} > 0,$$

so implied vol is unique wherever it exists and root-finding on a bracket is
the right tool. `implied_vol` first rejects prices outside the static bounds
$[\max(Se^{-qT} - Ke^{-rT},\,0),\; Se^{-qT}]$, then runs Brent's method
(1973): inverse quadratic interpolation, a bisection fallback, and the
standard guard on step size, so it keeps superlinear convergence without ever
losing the bracket.

Vega collapses as $\sigma \to 0$ and as options move deep in or out of the
money; a Newton iteration will happily diverge there, which is why this is a
bracketed solver rather than a Newton one. If the vol lies outside
$[10^{-6},\,5]$ the error says which end it fell off, because for short-dated
or event-driven names the upper bracket is a real constraint and not a
theoretical one.

## 4. SABR — `sabr.py`, `sabr_fit.py`

$$\mathrm{d}F = \alpha F^\beta\,\mathrm{d}W_1, \qquad
\mathrm{d}\alpha = \nu \alpha\,\mathrm{d}W_2, \qquad
\mathrm{d}W_1\,\mathrm{d}W_2 = \rho\,\mathrm{d}t .$$

Hagan, Kumar, Lesniewski and Woodward (2002) give a singular-perturbation
expansion for the Black implied vol of this model,

$$\sigma_B(F, K) = \frac{\alpha}
{(FK)^{(1-\beta)/2}\left[1 + \frac{(1-\beta)^2}{24}\ln^2\frac{F}{K}
+ \frac{(1-\beta)^4}{1920}\ln^4\frac{F}{K}\right]}
\cdot \frac{z}{x(z)} \cdot \Big[1 + (\cdots)\,T\Big],$$

$$z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln\frac{F}{K}, \qquad
x(z) = \ln\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}.$$

**The at-the-money trap.** At $K = F$ we have $z = 0$ and $x(z) = 0$, so
$z/x(z)$ is $0/0$. The usual shortcut is to branch: below some tolerance on
$|K - F|$, return the ATM formula. That is where implementations go wrong, in
two ways. The branch typically drops the $[1 + (\cdots)T]$ factor, and even
when it does not, there is a band of strikes just outside the tolerance where
$x(z)$ is the logarithm of a number extremely close to $1$ and loses most of
its significant digits.

The fix is that $z/x(z)$ is analytic at the origin. Expanding
$x(z) = z - \tfrac{1}{2}\rho z^2 + \tfrac{1}{6}(3\rho^2-1)z^3 + O(z^4)$ and
inverting the ratio,

$$\frac{z}{x(z)} = 1 + \frac{\rho z}{2}
+ \frac{3\rho^2 - 2}{12}\,z^2 + O(z^3).$$

`sabr_iv` evaluates this series for $|z| < 10^{-6}$ and the closed form
otherwise, applying the time correction on both paths. The at-the-money value
is then the limit of one formula rather than a special case, and
`examples/fit_sabr.py` shows the vol converging to it at the expected linear
rate in $z$ — the gap falls by a factor of 100 for every two decades $K$
moves towards $F$.

Calibration fixes $\beta$ — a scale convention, not identifiable from one
slice, since $\beta$ and $\rho$ trade off — and fits $(\alpha, \rho, \nu)$ by
weighted least squares from a coarse grid start.

## 5. SVI — `svi.py`

Gatheral's raw parameterisation (2004) models *total implied variance*
$w(k) = \sigma_{\text{iv}}^2(k)\,T$ against log-moneyness $k = \ln(K/F)$:

$$w(k) = a + b\Big(\rho\,(k - m) + \sqrt{(k-m)^2 + \sigma^2}\Big).$$

Five parameters, each with a job: $a$ sets the level, $b$ the overall angle
between the wings, $\rho$ the tilt between them, $m$ the horizontal position
of the minimum, $\sigma$ how rounded the elbow is. The wings are
asymptotically linear in $k$ with slopes $b(1 \mp \rho)$ — exactly the shape
Lee's moment formula (2004) says an arbitrage-free smile must have, which is
the reason for this functional form rather than a spline.

Total variance is the natural coordinate. It is the quantity that has to be
monotone across maturities, it is linear in $T$ for a flat surface, and it
stays finite at the short end where $\sigma_{\text{iv}}$ would not.

## 6. Butterfly arbitrage: the $g(k)$ function — `svi.py`

Rewriting Breeden–Litzenberger in log-moneyness expresses the density of $k$
directly in terms of the slice. Gatheral and Jacquier (2014) show that for
any twice-differentiable $w$ with $w > 0$,

$$p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}\,
\exp\!\left(-\frac{d_-(k)^2}{2}\right), \qquad
d_-(k) = -\frac{k}{\sqrt{w(k)}} - \frac{\sqrt{w(k)}}{2},$$

$$g(k) = \left(1 - \frac{k\,w'(k)}{2\,w(k)}\right)^{\!2}
- \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac14\right)
+ \frac{w''(k)}{2}.$$

The Gaussian factor is strictly positive, so **the slice is free of butterfly
arbitrage if and only if $g(k) \ge 0$ everywhere**. Reading the three terms:
the first is the Jacobian of the change of variables from strike to
moneyness, the second is what a sloped smile costs, the third is the
convexity that has to pay for it. A smile can be smooth, monotone in the
wings and visually unremarkable and still fail, because the failure lives in
the balance of those terms rather than in the shape of the curve. That is
what the README's lead figure shows.

For raw SVI both derivatives are closed-form. With $y = k - m$ and
$r = \sqrt{y^2 + \sigma^2}$,

$$w' = b\left(\rho + \frac{y}{r}\right), \qquad
w'' = \frac{b\,\sigma^2}{r^3},$$

so `svi_g` and `svi_density` are exact rather than finite differences. Note
that $w'' > 0$ always: raw SVI is convex in $k$ and still admits butterfly
arbitrage. Convexity of the smile is not the condition.

`svi_min_g` scans $g$ on a grid and returns the minimum *and where it
occurs*, since knowing which strikes are implicated is the point of running
the check. `fit_svi_slice` runs that scan on its own output and warns rather
than returning an inadmissible slice in silence.

## 7. Calendar arbitrage — `noarb.py`, `surface.py`

For $T_1 < T_2$ a calendar spread cannot be worth less than nothing, which in
total-variance coordinates is the clean statement

$$w(k, T_1) \le w(k, T_2) \quad \text{for every } k$$

(Gatheral 2004 — in $k$, not in strike, because the forward moves with $T$).
`SVISurface` interpolates $w$ linearly in $T$ between fitted slices, which
preserves the ordering for free: if the slices are ordered at $k$, so is
every point between them. Below the front expiry $w$ is scaled proportionally
to $T$ rather than held flat, because holding $w$ flat sends
$\sigma_{\text{iv}} = \sqrt{w/T} \to \infty$ as $T \to 0$, which is not a
boundary condition anyone wants.

## 8. Quasi-explicit calibration — `surface.py`

Fitting raw SVI looks like a five-parameter least-squares problem, and
treated as one it fails. The objective has a deep local minimum at
$\sigma \to 0$ with $|\rho| \to 1$: the elbow collapses to a corner that
threads the middle of the data while both wings go wrong. On a steep
single-name skew a single-start Nelder-Mead walks straight into it from a
standard initial guess and reports convergence.

De Marco and Martini (Zeliade, 2009) observed that only two of the five
parameters enter non-linearly. Substituting $y = (k-m)/\sigma$,

$$w = a + \underbrace{b\rho\sigma}_{d}\;y
      + \underbrace{b\sigma}_{c}\;\sqrt{y^2 + 1},$$

which for fixed $(m, \sigma)$ is **linear** in $(a, d, c)$. The inner problem
is a $3\times3$ normal-equations solve, done exactly; the SVI constraints
$b \ge 0$ and $|\rho| < 1$ become the cone $c \ge 0$, $|d| \le c$, and the
intercept is refitted after any projection onto it. Only the outer problem in
$(m, \sigma)$ needs searching, and two dimensions are cheap to cover with a
coarse grid followed by local descent from the best few starts.

`fit_svi_slice` does exactly that, then polishes in the full 5-D space from
the quasi-explicit solution and keeps the polish only if it improved the
loss. On the skew that defeats the naive fit the reduction reaches the global
minimum to machine precision; the second README figure is that comparison.

When `butterfly_penalty > 0`, the polish objective gains
$\lambda \sum_k \min(g(k), 0)^2$ over the fit range. This is a soft
constraint rather than a projection onto the arbitrage-free set: it buys
non-negativity of the density with least-squares error, and the README's lead
figure quantifies the exchange rate on the standard pathological slice.

## 9. What is deliberately not here

- **Calendar *enforcement*.** Slices are fitted independently and the
  calendar condition is checked afterwards. Enforcing it during calibration
  needs a fit coupled across expiries; Gatheral–Jacquier's SSVI, whose
  parameterisation is arbitrage-free by construction, is the usual route.
- **Local volatility.** Dupire's $\sigma_{\text{loc}}$ follows from $w(k,T)$
  and reuses the same $g$-like denominator, but it needs
  $\partial w/\partial T$ to be smoother than the piecewise-linear
  interpolation used here.
- **Vectorisation.** Everything is scalar Python. A $50\times20$ grid is a
  few thousand calls, which is fine for calibration and not fine inside a
  Monte Carlo loop.

## References

1. D. Breeden, R. Litzenberger (1978). *Prices of State-Contingent Claims
   Implicit in Option Prices.* Journal of Business 51(4), 621–651.
2. R. Brent (1973). *Algorithms for Minimization without Derivatives*,
   chapter 4. Prentice-Hall.
3. P. Hagan, D. Kumar, A. Lesniewski, D. Woodward (2002). *Managing Smile
   Risk.* Wilmott Magazine, September, 84–108.
4. R. Lee (2004). *The Moment Formula for Implied Volatility at Extreme
   Strikes.* Mathematical Finance 14(3), 469–480.
5. J. Gatheral (2004). *A Parsimonious Arbitrage-Free Implied Volatility
   Parameterization with Application to the Valuation of Volatility
   Derivatives.* Global Derivatives & Risk Management, Madrid.
6. S. De Marco, C. Martini (2009). *Quasi-Explicit Calibration of Gatheral's
   SVI Model.* Zeliade Systems white paper ZWP-0005.
7. J. Gatheral, A. Jacquier (2014). *Arbitrage-Free SVI Volatility Surfaces.*
   Quantitative Finance 14(1), 59–71.
