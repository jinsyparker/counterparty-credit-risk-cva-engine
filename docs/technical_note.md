# Technical note: models, formulas and validation

This note documents every model choice in the engine and the formula behind it, so each number in the reports can be traced back to an equation and a test. Results are in `reports/final_project_summary.md`; the code paths are named in each section.

## 1. Scope and pipeline

The engine estimates future counterparty exposure and **unilateral CVA** for a small portfolio of vanilla USD fixed-for-floating interest-rate swaps:

Market data → swap valuation → rate simulation → future MtM → exposure → default probability → CVA → netting / collateral → report

Out of scope by design: exotics, multi-curve pricing, bilateral CVA/DVA/FVA/MVA, stochastic credit, wrong-way risk, initial margin, ML surrogates.

## 2. Conventions

| Item | Convention |
|---|---|
| Time | Year fractions from the valuation date (2 Oct 2026). Accrual = time difference; no day counts, calendars or business-day rolls. |
| Rates | Decimals; zero rates continuously compounded, P(0,t) = e^(−z(t)t). |
| Sign | Values from our (the bank's) side: V = PV(received) − PV(paid). A payer swap pays fixed. |
| Cash-flow timing | V(t) excludes cash flows paid at t. Every reset and payment date is on the simulation grid. |
| Random numbers | `np.random.default_rng(seed)` passed explicitly; base seed 42; sensitivities reuse the seed (common random numbers). |

## 3. Yield curve (`src/curves.py`)

**Input.** The US Treasury daily par-yield curve (13 tenors, 1M–30Y), stored as a dated CSV snapshot.

**Bootstrap.** Each quote is treated as the coupon of a semiannual bond priced at par:

  1 = y · Σₖ τₖ P(0,Tₖ) + P(0,Tₙ)

Tenors of up to six months are a single payment, P(0,T) = 1/(1 + yT). Beyond six months, par yields are interpolated linearly onto every coupon date and the discount factors are solved one date at a time:

  P(0,Tₙ) = (1 − yₙ Σₖ₍ₖ<ₙ₎ τₖ P(0,Tₖ)) / (1 + yₙ τₙ)

The resulting curve reprices every input bond to par (tested to 1e-10).

**Interpolation.** Linear in log P(0,t), which means piecewise-constant instantaneous forwards. This reproduces the nodes, keeps discount factors positive, and keeps them decreasing whenever forwards are positive. Beyond the last node the last forward is held flat.

**Known artifact.** Linear par-yield interpolation produces a sawtooth in the forwards, and a jump at 20Y because the 20Y Treasury yields more than the 30Y on the snapshot date. Every trade matures within 10Y, where forwards are smooth by comparison.

## 4. Swap pricing (`src/swap_pricer.py`)

Schedules are regular and generated backward from maturity (fixed semiannual, floating quarterly), with a short front stub if needed. With notional N, accruals τ and discount factors P:

| Quantity | Formula |
|---|---|
| Annuity | A = Σᵢ τᵢ P(Tᵢ) |
| Fixed leg | PV_fixed = N K A |
| Forward rate | Fⱼ = (P(Sⱼ)/P(Tⱼ) − 1)/τⱼ |
| Floating leg | PV_float = N Σⱼ τⱼ Fⱼ P(Tⱼ) = N (P(T₀) − P(Tₙ)) in a single curve |
| Par rate | K_par = PV_float / (N A) |
| Value | V = sign · (PV_float − PV_fixed) |

With one curve and semiannual fixed coupons, the par swap rate equals the par bond yield, (1 − P(0,T))/A. The 10Y swap reprices exactly to the 10Y Treasury par yield, which is a free consistency check.

The leg functions take **arrays of discount factors** with cash flows on the last axis. The same code therefore prices at t = 0 (curve discount factors) and on simulated paths (Hull-White P(t,T) with shape paths × cash flows).

## 5. Hull-White one-factor model (`src/rate_models.py`)

**Dynamics.** dr = (θ(t) − a r) dt + σ dW, with θ(t) chosen so the model reproduces today's curve. Write r(t) = x(t) + φ(t), where

  dx = −a x dt + σ dW, x(0) = 0,  φ(t) = f(0,t) + σ²/(2a²) (1 − e^(−at))²

**Bond prices.** With B(t,T) = (1 − e^(−a(T−t)))/a and V_x(t) = Var x(t) = σ²(1 − e^(−2at))/(2a):

  P(t,T) = P(0,T)/P(0,t) · exp(−B x(t) − ½ B² V_x(t) − B σ²/(2a²) (1 − e^(−at))²)

The instantaneous forward f(0,t) cancels, so revaluation needs only the discount curve and the simulated x(t), and the model fits today's curve automatically. At t = 0, P(0,T) is reproduced exactly (tested).

**Exact simulation.** x is Ornstein-Uhlenbeck, so over a step Δ, (x, I = ∫x ds) has an exact joint Gaussian transition:

  x(t+Δ) = x(t) e^(−aΔ) + ε_x,  I(t+Δ) − I(t) = x(t) B(Δ) + ε_I

with Var ε_x = σ²(1 − e^(−2aΔ))/(2a), Var ε_I = σ²/a² (Δ − 2B(Δ) + (1 − e^(−2aΔ))/(2a)) and Cov(ε_x, ε_I) = σ²B(Δ)²/2. Shocks are drawn through the 2×2 Cholesky factor. There is no discretization bias, so the grid can be irregular.

**Stochastic discount factor.** D(0,t) = exp(−∫₀ᵗ r ds) = P(0,t) · exp(−I(t) − ½ Var I(t)), a lognormal martingale correction with E[D(0,t)] = P(0,t).

**Parameters.** a = 3% per year and σ = 1%/√year, the absolute (normal) diffusion volatility of the short rate. σ is a model parameter, not a quoted swaption normal (Bachelier) volatility. Both are assumed, not calibrated to swaptions; Section 12 reports sensitivities to both.

## 6. Revaluation (`src/simulation.py`)

The grid is monthly out to the last maturity, merged with every effective, reset and payment date (121 dates). At each date t:

1. P(t,T) is computed once for every future portfolio date (paths × dates).
2. A floating coupon whose accrual starts at t fixes on each path at L = (1/P(t,T) − 1)/τ. That fixing replaces the forward rate while the coupon is running.
3. Each trade's remaining legs are priced with the time-0 leg formulas.

The result is the MtM cube V[trade, path, date]. Tests check that the cube at t = 0 equals the time-0 pricer. They also check that with σ = 0 every path equals the deterministic forward value to 1e-9, including mid-coupon dates, which exercises the fixing logic.

## 7. Exposure (`src/exposure.py`)

| Quantity | Definition |
|---|---|
| Netted exposure | E_net(t) = max(Σᵢ∈NS Vᵢ(t), 0), with netting applied on each path and date **before** the positive part |
| Gross exposure | E_gross(t) = Σᵢ max(Vᵢ(t), 0) ≥ E_net(t), because the positive part is subadditive |
| Expected Exposure | EE(t) = mean over paths of E(t) |
| PFE | PFE_q(t) = min{e : Pₙ(E(t) ≤ e) ≥ q}, the empirical inverse CDF, so PFE is an actual simulated value |
| EPE (time-average EE) | (1/T) ∫₀ᵀ EE(t) dt (trapezoid, T = netting set's last maturity) |

"EPE" always means this life-time average. It is **not** the regulatory Effective EPE, which averages the running maximum of EE over the first year only.

## 8. Credit (`src/credit.py`)

Default is the first jump of a Poisson process with deterministic intensity λ(t), piecewise constant:

  S(t) = exp(−∫₀ᵗ λ), PD(t) = 1 − S(t), ΔPDᵢ = S(tᵢ₋₁) − S(tᵢ), LGD = 1 − R

Spreads are **assumed counterparty spread scenarios (CDS-equivalent)**, not market quotes. Hazards use the credit-triangle approximation λ ≈ s/(1 − R). It ignores premium accrual, discounting inside the CDS legs and the spread term structure, so it is an approximation, not a calibration to a CDS curve. Marginal PDs over a grid telescope, so they sum exactly to PD at the last date.

## 9. CVA (`src/cva.py`)

With default independent of rates:

  CVA = LGD ∫₀ᵀ E[D(0,t) E(t)] dPD(t) ≈ LGD Σᵢ ½ [dEE(tᵢ₋₁) + dEE(tᵢ)] ΔPDᵢ

- **Headline:** dEE(t) = E[D(0,t) E(t)], discounted pathwise with the simulated bank account. Exposure and discount factors are correlated under stochastic rates: these books gain when rates rise, which is exactly when D is small.
- **Benchmark:** the spec §7 approximation dEE(t) = P(0,t) EE(t). Because E[D(0,t)] = P(0,t), the difference is exactly the covariance between D and E. The two coincide when σ = 0 (tested).
- **Integration grid:** the sum runs over all 120 intervals of the simulation grid. Yearly buckets only regroup interval contributions for presentation.
- **Error bars:** the estimator is linear in exposure, so CVA equals the average of a per-path CVA, and the standard error is std/√N. The portfolio total's error comes from the per-path sum over netting sets: they share paths, so their errors are correlated and cannot be added in quadrature.

## 10. Collateral (`src/collateral.py`, `src/simulation.py::refine_paths`)

A simplified **one-way CSA**: only the counterparty posts, consistent with unilateral CVA. If it defaults at t, the last call it met was at t − MPoR:

  C(t) = max(V(t − MPoR) − H − MTA, 0),  E_c(t) = max(V(t) − C(t), 0)

For t < MPoR the look-back uses V(0). Properties (all tested):
- C ≥ 0, so E_c ≤ max(V, 0).
- E_c rises with H and MTA on every path.
- H = ∞ gives the uncollateralized exposure exactly.
- H = 0 with MPoR = 0 gives zero exposure.

**Look-back dates on the same paths.** V(t − MPoR) is needed between grid dates. Re-simulating on a finer grid would change every random draw, so the uncollateralized CVA would no longer equal the Phase D headline. Instead, each new date s is drawn from the exact conditional law of (x(s), I(s)) given the nearest known state on the left and the next original state on the right:

  Y | left ~ N(μ_Y, S₁),  Z = (x_R, I_R) = A Y + noise ~ N(0, S₂),  A = [[e^(−aΔ₂), 0], [B(Δ₂), 1]]

  Y | Z ~ N(μ_Y + K (Z − A μ_Y), S₁ − K A S₁),  K = S₁ Aᵀ (A S₁ Aᵀ + S₂)⁻¹

(x, I) is Markov and Gaussian, so the refined paths have exactly the joint law of a fine-grid simulation, and every original date is bit-identical. Tests check the marginal variances, the autocorrelation and the martingale property at bridged dates.

**Simplifications.**
- MTA is added to the threshold.
- Both parties keep paying coupons during the MPoR. A coupon we pay inside the window therefore raises exposure, producing spikes on our net payment dates.
- No independent amount, initial margin, haircuts or disputes.

## 11. Validation strategy (`tests/`)

- **Exact identities** use tolerances of 1e-9 or tighter: par swap = 0, payer = −receiver, curve node reproduction, bootstrap repricing, σ = 0 forward values, CVA zero cases, CVA linearity in LGD and notional, and the continuous closed form for constant inputs.
- **Monte Carlo statements** use fixed seeds and a 4-standard-error tolerance that is never tuned to pass: analytic Hull-White moments, mean-reversion slope, the martingale E[D(0,t)P(t,T)] = P(0,T), swap martingales, and 1/√N error decay.
- **Monotonicity** checks reuse the same paths, so they hold exactly rather than statistically: netted ≤ gross, tighter collateral ⇒ lower exposure, CVA monotone in hazard, recovery, notional and threshold.
- **If a statistic looks off,** rerun with many more paths before deciding. One mean-reversion regression was 3.1 standard errors out at 10,000 paths and well within one standard error at 400,000.

## 12. Sensitivities and convergence (`src/validation.py`)

- **Common random numbers.** Every market sensitivity re-simulates with the base seed. A σ change on the same grid reuses identical shocks, because x scales linearly with σ.
- **Tenor.** Each trade's tenor is multiplied by a factor (rounded to a quarter) and the trade is re-struck at par plus its original offset, so the maturity effect is not mixed with a change in moneyness.
- **Credit.** Recovery is reported two ways. With the hazard fixed, CVA ∝ 1 − R. With the spread fixed, the hazard is re-implied and LGD × λ ≈ s, so CVA barely moves.
- **Convergence.** Independent 10,000-path chunks are pooled (chunk k uses seed 42 + 1000k). Chunk 0 is the base case, so the 10,000-path point equals the headline exactly.

## 13. Limitations and model risk

| Limitation | Likely effect |
|---|---|
| Single-curve Treasury pricing (no SOFR/OIS basis) | Levels shift slightly; netting and collateral conclusions are unaffected |
| One-factor Gaussian Hull-White | All rates move together, so curve-hedged books (NS_A) look less risky than they are; rates can go negative (about 1% of paths at 10Y) |
| Assumed, uncalibrated a and σ | CVA scales roughly linearly with σ for directional books (see sensitivities) |
| Credit-triangle hazards from assumed spreads | Approximate default probabilities; recovery effects partly cancel when spreads are fixed |
| Deterministic hazards independent of rates | No wrong-way risk: CVA is understated if the counterparty's credit worsens when we are owed money |
| Unilateral CVA only | No DVA or funding adjustments |
| Simplified CSA | No initial margin or haircuts; the cash-flow-spike treatment is one of several market conventions |
| No day counts, calendars or historical fixings | Small cash-flow timing differences versus production systems |

## 14. Natural extensions

Swaption calibration of a and σ; a two-factor model (G2++) to capture curve risk; Jamshidian closed-form swaption prices as an exposure benchmark; wrong-way risk through correlated stochastic hazards; bilateral CVA/DVA; SOFR multi-curve.
