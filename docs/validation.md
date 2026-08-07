# What has been checked against what

A library that only tests itself against itself tells you it is
self-consistent. This page is the other kind of evidence: every row below
compares `volsurf` to something it does not control: a closed form written
out in the source paper, a limit the model has to collapse to, a Monte-Carlo
simulation of the SDE the formula is only an approximation *of*, or a second
screen built on a different principle.

Everything here is printed by

```bash
PYTHONPATH=. python examples/validate.py
```

and asserted by `tests/test_validation.py`, which imports the same reference
implementations at smaller sample sizes. The numbers below are pasted from a
run; nothing was rounded to make it look better, and the places where the
library and the reference genuinely disagree have a section of their own
rather than a footnote.

---

## The table

| Claim | Our value | Reference value | Source of the reference |
| --- | --- | --- | --- |
| `price(call) - price(put)` is the forward | max discrepancy `1.067e-15 · S` over 5,000 random `(S, K, T, r, q, σ)` | exactly `S e^{-qT} - K e^{-rT}` | put-call parity, static replication, no model |
| Call price sits inside the static bounds | max violation `1.333e-16 · S` | `max(F-K,0) ≤ C ≤ S e^{-qT}` | no-arbitrage bounds |
| `implied_vol(price(σ)) = σ` | max error `1.845e-10` | the σ it was given | round trip |
| `σ → 0` call value | `11.664885355551064` | `11.664885355551064` | discounted intrinsic `max(F-K,0)e^{-rT}` |
| `σ → ∞` call value | `99.0049833749168` | `99.0049833749168` | `S e^{-qT}` |
| `∂²C/∂K²` at K=105 | `0.014808613003` | `0.014808614212` (rel `8.2e-08`) | Breeden & Litzenberger (1978): `∂²C/∂K² = e^{-rT} q(K)`, `q` the lognormal density |
| SABR at `β=1, ν=0` | `max abs err = 0.0` over 18 `(K,T)` | exactly `α`, because the model *is* Black there | Hagan, Kumar, Lesniewski & Woodward (2002), the SABR SDE |
| SABR at the money | max rel err `2.557e-15` over 2,000 random parameter sets | `σ_B(f,f)` transcribed from the paper | Hagan et al. (2002), the at-the-money case of their implied-vol expansion |
| SABR at `β=0, ν→0` | `0.022314355140` at K=80 | `0.022314355131` (rel `3.8e-10`) | `α log(F/K)/(F-K)`; the O(log⁶) residual is the paper's own truncation |
| SABR vs the SDE, `ν²T = 0.16` | Hagan − MC = `+1.5 bp` ATM, `+11.9 bp` at K=70 | Monte-Carlo of the SABR SDE, s.e. `1.0 bp` / `11.6 bp` | conditional Monte Carlo, see below |
| SABR vs the SDE, `ν²T = 3.2` | Hagan − MC = `+393 bp` ATM, `+1121 bp` at K=50 | same simulation, s.e. `2.2 bp` / `4.8 bp` | **disagrees, see "Where we differ"** |
| SVI with `b = 0` is Black-Scholes | max abs err `2.220e-16` | `φ(d_-)/√w`, the lognormal density in log-moneyness | Black-Scholes closed form |
| The implied density is a unit mass | `∫p dk - 1 = 0.0`, `∫e^k p dk - 1 ≤ 1.1e-16` | `1` and `1` | normalisation and `E[S_T] = F` |
| `g(k)` far in the wing | max err `3.0e-08` over 500 draws | `1/4 - b²(1+ρ)²/16` | derived below; non-negative iff the wing slope `b(1+ρ) ≤ 2`, which is Lee (2004)'s moment-formula bound |
| Wing slope above 2 forces `g < 0` | `300 / 300` random slices | all of them | contrapositive of the row above |
| SSVI inside the Gatheral-Jacquier conditions | min `g(k)` over 500 near-boundary draws = `+0.003137`, violations `0` | `g(k) ≥ 0` | Gatheral & Jacquier (2014), the sufficient conditions `θφ(1+\|ρ\|) < 4` and `θφ²(1+\|ρ\|) ≤ 4` for an SSVI slice to be butterfly-free |
| Analytic and discrete butterfly screens | `g < 0` on `k ∈ [0.642, 1.256]`; price screen flags `k ∈ [0.650, 1.250]` | each other, sharing no code | Vogt's slice at 5%-spaced quotes; the two agree to within one quote spacing |
| The surface's calendar check | worst decrease in `w` over a 241×400 `(k,T)` grid = `0.000e+00` | brute force, off the fitted nodes | `calendar_arbitrage_free()` returns `True` |
| The surface reproduces its own slices | max `\|iv - fitted slice iv\|` = `1.841e-08` | the slices it was fitted to | round trip through `fit_svi_surface` |

---

## The Monte-Carlo reference

Hagan's formula is a singular-perturbation expansion. The thing it
approximates is a price, so the honest reference is a price computed from the
SDE without going anywhere near the expansion.

At `β = 1` the model is

```
dF = α F dW₁,    dα = ν α dW₂,    d⟨W₁,W₂⟩ = ρ dt
```

Split `dW₁ = ρ dW₂ + √(1-ρ²) dW⊥`. Conditional on the whole `α` path, `log F_T`
is Gaussian, so the call is a Black-Scholes call at

```
F_eff = F₀ exp(ρ (α_T - α₀)/ν - ρ² V / 2),   σ_eff = √((1-ρ²) V / T)
```

with `V = ∫₀ᵀ α_t² dt` and `∫ α dW₂ = (α_T - α₀)/ν`, the latter because `α` is a
driftless geometric Brownian motion. Conditioning on the volatility path this
way is the Romano & Touzi (1997) mixing argument. It removes all the variance
contributed by `W⊥`, which is why 200,000 antithetic paths price the wings to
under a basis point of vol. The `α` path is exact (lognormal increments), so
the only discretisation is the trapezoid rule for `V`, and refining it does
nothing:

```
step refinement at nu=0.40, T=1, K=100 (Hagan - MC, bp):
     125 steps     +1.50
     250 steps     +1.51
     500 steps     +1.42
    1000 steps     +1.57
```

The comparison at three values of `ν²T`, the expansion's small parameter:

```
alpha=0.20 beta=1 rho=-0.30 nu=0.40 T=1.0   (nu^2 T = 0.16)
     K    MC vol     Hagan vol   Hagan - MC (bp)   MC s.e. (bp)
    70.0  0.232001   0.233186        +11.85          11.55
    85.0  0.212991   0.213567         +5.76           3.06
   100.0  0.200956   0.201107         +1.51           1.03
   115.0  0.195219   0.195150         -0.69           0.53
   130.0  0.194459   0.194349         -1.09           0.52

alpha=0.20 beta=1 rho=-0.30 nu=0.60 T=2.0   (nu^2 T = 0.72)
     K    MC vol     Hagan vol   Hagan - MC (bp)   MC s.e. (bp)
    60.0  0.276145   0.288551       +124.06           7.79
    80.0  0.230900   0.237317        +64.17           2.15
   100.0  0.204398   0.206780        +23.82           1.07
   125.0  0.199744   0.201284        +15.40           1.37
   160.0  0.219474   0.223937        +44.62           1.82

alpha=0.20 beta=1 rho=-0.30 nu=0.80 T=5.0   (nu^2 T = 3.20)
     K    MC vol     Hagan vol   Hagan - MC (bp)   MC s.e. (bp)
    50.0  0.299660   0.411751      +1120.91           4.82
    75.0  0.233165   0.300457       +672.92           2.03
   100.0  0.194811   0.234133       +393.23           2.24
   150.0  0.210681   0.263858       +531.77           2.81
   200.0  0.246354   0.326147       +797.93           3.19
```

## The wing limit of `g`, worked out

Raw SVI's wings are linear: as `k → +∞`, `w ≈ b(1+ρ)k`, `w' → b(1+ρ)` and
`w'' → 0`. Put that into

```
g(k) = (1 - k w'/(2w))² - (w'/2)²(1/w + 1/4) + w''/2
```

The first bracket tends to `1 - 1/2 = 1/2`, so its square tends to `1/4`. The
`1/w` inside the second term vanishes, leaving `b²(1+ρ)²/16`. So

```
g(+∞) = 1/4 - b²(1+ρ)²/16,     g(-∞) = 1/4 - b²(1-ρ)²/16
```

which is non-negative exactly when the wing slope is at most 2, the bound
Lee's (2004) moment formula puts on the asymptotic slope of total variance.
The library's `svi_min_g` is a local grid scan, and it is reassuring that it
agrees with the global asymptotic statement instead of contradicting it: 300
out of 300 random slices with `b(1+|ρ|) > 2` are caught.

A raw-SVI slice is an SSVI slice exactly when `a = b σ √(1-ρ²)`, with
`θ = 2bσ/√(1-ρ²)` and `φ = √(1-ρ²)/σ`. Mapping Gatheral and Jacquier's
sufficient conditions through that substitution and drawing 500 slices at
90–99.9% of the first bound gives a minimum `g` of `+0.003137` and no
violations, tight enough that the check is doing work and on the right side
of zero every time.

---

## Where we differ

**Hagan's formula is not the SABR model, and at large `ν²T` the difference is
enormous.** The third block above is the honest picture: at `ν²T = 3.2` the
expansion overstates the 50-strike vol by eleven vol points. Nothing is wrong
with the transcription. The same code is exact to `2.6e-15` against the
paper's own at-the-money formula. It is the expansion that has left its
regime, and `sabr_iv` reproduces the expansion, faithfully, including where
the expansion is wrong. If you are calibrating five-year options with a
vol-of-vol near 1, this library will not warn you, and it should not be the
tool you reach for.

**We ship Hagan's `z`, not Obłój's `ζ`.** Obłój (2008) points out that

```
z = (ν/α)(FK)^((1-β)/2) log(F/K)
```

is itself an approximation to

```
ζ = (ν/α)(F^(1-β) - K^(1-β))/(1-β)
```

with the two agreeing to leading order in `(1-β)log(F/K)`. `volsurf`
implements the original. What that costs, at `β = 0.5`:

```
      K    ships (z)   Obloj (zeta)   difference (bp of vol)
     50.0   0.307334     0.307740          -4.06
     70.0   0.251809     0.251859          -0.51
    100.0   0.201790     0.201790          +0.00
    140.0   0.181318     0.181332          -0.14
    200.0   0.191312     0.191613          -3.02
```

Four basis points of vol in the far wing at a two-to-one strike ratio, and
zero at the money. That is small next to the expansion error above, which is
why the original is still what ships. It is a real, documented
difference from a reference, and it belongs here rather than in a changelog
nobody reads.

**Total mass is not an arbitrage test.** The Vogt slice integrates to exactly
`1` and prices the forward to `1.1e-16`, and it still admits butterfly
arbitrage. A density with a hole in it and a compensating bump elsewhere
normalises perfectly well. That row in the table is evidence that the density
formula is right, not evidence that the slice is admissible; only `g(k) ≥ 0`
is that.

**What is not validated here.** The SVI fitter's behaviour on *noisy* data is
checked only for self-consistency, because there is no external benchmark for "the
right SVI parameters given noisy quotes", because there is no right answer.
The comparison against the SDE is done at `β = 1`, where the conditional law
of `F_T` is exactly lognormal; for `β ≠ 1` the conditional law is CEV and
pricing it needs a non-central chi-square, which is not in this library, so
those cases rest on the closed-form limits only. And there is no comparison
against a commercial pricer, because I do not have one to compare against.

---

## References

1. D. Breeden, R. Litzenberger (1978). *Prices of State-Contingent Claims
   Implicit in Option Prices.* Journal of Business 51(4), 621–651.
2. P. Hagan, D. Kumar, A. Lesniewski, D. Woodward (2002). *Managing Smile
   Risk.* Wilmott Magazine, September, 84–108.
3. M. Romano, N. Touzi (1997). *Contingent Claims and Market Completeness in a
   Stochastic Volatility Model.* Mathematical Finance 7(4), the source of the
   conditioning argument used for the Monte-Carlo reference.
4. R. Lee (2004). *The Moment Formula for Implied Volatility at Extreme
   Strikes.* Mathematical Finance 14(3), 469–480.
5. J. Gatheral, A. Jacquier (2014). *Arbitrage-Free SVI Volatility Surfaces.*
   Quantitative Finance 14(1), 59–71.
6. J. Obłój (2008). *Fine-tune your smile: Correction to Hagan et al.* Short
   note; the ζ variable discussed above is its subject.
