# Project plan & progress

The build roadmap and review log for the Counterparty Credit Risk & CVA Engine. Formulas and model choices are documented in `docs/technical_note.md`; this file records what was built, in what order, and the changes made after review.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Rate model | Hull-White 1F, simulated exactly through x(t) = r(t) − φ(t) | Fits today's curve exactly, has closed-form P(t,T), and has no discretization bias |
| Curve | US Treasury par yields (2 Oct 2026 snapshot), bootstrapped; log-linear DF interpolation | Real public data; the bootstrap reprices its inputs exactly |
| Pricing | Single-curve; leg formulas on discount-factor arrays | The same code prices at t=0 and on simulated paths |
| CVA | Headline: pathwise-discounted LGD·Σ E[D(0,t)·E(t)]·ΔPD on the full grid; spec §7 P(0,t)·EE as a benchmark | Exposure and discount factors are correlated under stochastic rates (changed after the Phases C–D review) |
| Collateral | One-way CSA with threshold, MTA and MPoR look-back | Simple, but still leaves residual exposure under a zero threshold |
| Parameters | a = 0.03, σ = 1%/√yr short-rate diffusion vol (assumed, not a swaption vol), 10k paths, seed 42, PFE 95%, assumed CDS-equivalent spreads 150/300bp, R = 40% | Transparent assumptions, each with a sensitivity grid |

## Portfolio (`data/reference_inputs/portfolio.csv`)

10 swaps, USD, fixed semiannual / float quarterly. Fixed rates are par ± offset on the snapshot curve.

- **NS_A / CPTY_A (6 trades):** payers and receivers with offsetting maturities (A1 10Y payer 50m vs A4 10Y receiver 40m, …), including A3, a 1Y-into-5Y forward-starting swap. Netting should be very effective.
- **NS_B / CPTY_B (4 trades):** mostly payers, so netting should help little.

## Phases

### Setup ✅
- [x] git init, Python 3.11 venv (uv), requirements, `.gitignore`, `pytest.ini`
- [x] Project conventions, `docs/PLAN.md`, starter `README.md`

### Phase A — Foundations ✅
- [x] Treasury snapshot + `scripts/fetch_treasury_curve.py`
- [x] `src/curves.py`: `DiscountCurve`, `bootstrap_par_yields`
- [x] `src/instruments.py`: `Swap`, schedules, `load_portfolio`, `group_by_netting_set`
- [x] `src/swap_pricer.py`: leg formulas on DF arrays, par rate, MtM, cash-flow table, portfolio valuation
- [x] Synthetic portfolio (`scripts/make_synthetic_portfolio.py`) + `counterparties.csv`
- [x] `run_all.py` stage A, `notebooks/01_swap_pricing.ipynb`
- [x] Tests: curve (nodes, DF monotone, bootstrap repricing), instruments, pricing (par ≈ 0, symmetry, closed forms, aggregation)

### Phase B — Market simulation ✅
- [x] `src/rate_models.py`: `HullWhiteParams`, `bond_b`, `zcb_price`, exact joint (x, ∫x) transition, `short_rate`, `stochastic_discount`
- [x] `src/simulation.py`: `build_time_grid` (monthly + all pay and reset dates), `simulate_paths`, `revalue_portfolio` → MtM cube, `netting_set_mtm`
- [x] `src/config.py`: simulation base case; `run_all.py --quick`; `src/validation.py` check tables
- [x] Tests: E[x] = 0, Var[x], Var[∫x] and Cov analytic; mean-reversion slope; t=0 reproduces curve; martingale E[D(0,t)P(t,T)] ≈ P(0,T); forward-starting and spot swap martingales; σ=0 values equal forward values; cube at t=0 equals the Phase A pricer; a missing reset date raises
- [x] Notebook 02: paths, fan chart, short-rate distributions, checks, netting-set MtM fans

### Phase C — Exposure ✅
- [x] `src/exposure.py`: positive exposure, EE, PFE(q) (empirical inverse CDF), EPE, netted and gross exposure per netting set, summary and profile tables
- [x] Tests: E ≥ 0; netted ≤ gross pathwise; PFE reconciles exactly with the empirical CDF; EE(0) = max(V0, 0); known N(0,1) case; σ = 0 case; single-trade netting set has no benefit
- [x] Notebook 03: EE/PFE profiles, exposure distributions at 1Y/3Y/5Y, gross vs netted
- [x] `run_all.py` stage C: `exposure_profiles.csv`, `exposure_summary.csv`, three figures

### Phase D — Credit & CVA ✅
- [x] `src/credit.py`: `HazardCurve` (constant or piecewise), survival, cumulative PD, marginal PD, `hazard_from_spread`, counterparty loader
- [x] `src/cva.py`: CVA by bucket, total, stochastic-discount variant, pathwise standard error, yearly aggregation
- [x] Tests: ΣΔPD = PD(T); CVA monotone in λ, R and notional; λ = 0 → 0; R = 1 → 0; netted ≤ gross; continuous closed form; SE shrinks ~1/√N (slow)
- [x] Credit-only sensitivities (spread; recovery with hazard fixed vs spread fixed) in `src/validation.py`
- [x] `run_all.py` stage D and notebook 04 (CVA part; Phase F adds market sensitivities and convergence)

### Review changes after Phases C–D ✅ (2026-10-04)
- [x] HIGH: headline CVA is pathwise-discounted; curve-DF × EE kept as the benchmark
- [x] HIGH: confirmed CVA integrates over all 120 simulation-grid intervals (yearly buckets are presentation only); stated in output, docs and a test
- [x] MED: "EPE (time-average EE)" defined everywhere, not Effective EPE
- [x] MED: spreads described as assumed CDS-equivalent scenarios; λ ≈ s/(1−R) stated as an approximation
- [x] LOW: martingale wording separates the 10k base run from the 400k rerun
- [x] LOW: σ described as the short-rate diffusion volatility, not a swaption normal vol

### Phase E — Netting & collateral ✅
- [x] `src/collateral.py`: `CsaTerms`, collateral balance (H, MTA, MPoR look-back), collateralized exposure, `collateral_analysis`
- [x] `src/simulation.py::refine_paths`: exact Gaussian bridge adds the look-back dates to the Phase D paths, so the uncollateralized CVA equals the Phase D headline exactly
- [x] Gross vs netted vs collateralized EE and CVA with % reductions, per netting set; zero-threshold MPoR sensitivity (7 / 14 / 28 days)
- [x] Tests: lower threshold → weakly lower exposure pathwise and lower CVA; H = ∞ equals uncollateralized; E_c ≤ E; H = 0 and MPoR = 0 → 0; longer MPoR → more exposure; bridge marginals, autocorrelation and martingale
- [x] `run_all.py` stage E (4 tables, 2 figures) and `notebooks/05_collateral.ipynb`

### Phase F — Validation & final report ✅
- [x] `src/validation.py`: convergence (500 → 50k paths, nested pooled chunks, 95% CIs); sensitivities to σ, a, tenor (re-struck at par + offset) and notional with common random numbers; tornado including spread and recovery
- [x] Full `run_all.py` (stages A–F), notebook 04 sections 5–6
- [x] `docs/technical_note.md`, final `README.md` (assumptions, validation, limitations; results block generated), `reports/final_project_summary.md` with two résumé bullets (generated by `src/reporting.py`)
- [x] Final PDF report covering Phases A–F

### Optional extensions (only after F)
- [ ] G: Vasicek vs Hull-White model risk in EE/CVA
- [ ] A: wrong-way risk, or B: bilateral CVA/DVA

## Known limitations (feed into the README)

- Single-curve Treasury discounting and projection (no SOFR/OIS multi-curve).
- No day counts, calendars or business-day adjustments; all trades start on or after the valuation date.
- Hull-White parameters are assumed, not calibrated to swaptions.
- One factor: all rates are perfectly correlated, so curve-neutral books (NS_A) look nearly riskless. Gaussian rates can go negative (about 1% of paths at 10Y).
- Linear par-yield interpolation gives sawtooth forwards and a 20Y jump (a real 20Y/30Y inversion in the data).
- Credit-triangle hazards from assumed spreads (no CDS-curve calibration); deterministic hazard rates, independent of rates (no wrong-way risk); unilateral CVA only.
- Simplified collateral: one-way, MTA added to the threshold, both parties keep paying coupons during the MPoR (which creates exposure spikes on our net payment dates), no independent amount, initial margin or haircuts.
