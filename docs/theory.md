# Volatility surface math

## Black-Scholes recap

Risk-neutral call price under BS with continuous dividends:
\[
C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)
\]
with \(d_1 = (\ln(S/K) + (r - q + \sigma^2/2) T)/(\sigma \sqrt T)\) and
\(d_2 = d_1 - \sigma \sqrt T\).

## Implied volatility

Given a market price \(C^*\), implied vol solves \(C(\sigma) = C^*\). The
mapping is strictly monotone in \(\sigma\) on \((0, \infty)\) so Brent's method
on a bounded bracket is the standard tool.

## SABR (Hagan et al. 2002)

\[
\begin{aligned}
\mathrm{d}F &= \alpha F^\beta \,\mathrm{d}W_1 \\
\mathrm{d}\alpha &= \nu \alpha \,\mathrm{d}W_2 \\
\mathrm{d}W_1 \,\mathrm{d}W_2 &= \rho \,\mathrm{d}t
\end{aligned}
\]

Hagan derived a closed-form implied vol approximation \(\sigma_B(F, K, T)\).
We fix \(\beta\) (commonly 0.5 for FX/IR, 1.0 for equity index) and calibrate
\((\alpha, \rho, \nu)\).

## SVI raw (Gatheral 2004)

Total implied variance is parameterized as
\[
w(k) = a + b\left(\rho (k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)
\]
with \(k = \ln(K/F)\). Implied vol is \(\sigma_{iv}(k) = \sqrt{w(k)/T}\).

## No-arbitrage

- **Butterfly** (static, fixed expiry): the BS-implied density is
  \(\partial^2 C / \partial K^2 \geq 0\). Numerical violations indicate the
  smile crosses into an arbitrageable region.
- **Calendar**: total implied variance \(w(k, T)\) must be non-decreasing in
  \(T\) at each \(k\); otherwise a calendar spread is risk-free positive.

## References

- Hagan, Kumar, Lesniewski, Woodward (2002), *Managing Smile Risk*, Wilmott.
- Gatheral (2004), *A parsimonious arbitrage-free implied volatility
  parameterization*.
- Gatheral, Jacquier (2014), *Arbitrage-free SVI volatility surfaces*.
