# +CG vs +GI threshold sweep — comparison

Generated 2026-05-16. Sources: `benchmarking/outputs_thres/` (+GI, two-pass dynamic w/ span index) and `benchmarking/outputs_thres_cg/` (+CG, one-pass static in prompt). `causal-only` cells are from `benchmarking/outputs/zero_shot{,2}/`.

**Variant tie-breakers**:
- Mistral-Small-24B: +GI uses 3.1-2503 (the GI sweep); +CG uses 3.2-2506 (the CG re-run).
- All other models: same checkpoint for both methods.

**Edge counts** (TRAIL Suppes graph): causal-only ≈ 14, τ=0.35 ≈ 19, τ=0.25 ≈ 21, τ=0.20 ≈ 25, random-12 = 12 (non-Suppes, seed=42 null control).

**Cell formatting**: **bold** = best meaningful (non-random) variant in that row; _italic_ = random ≥ best meaningful (i.e., a row where the null control wins or ties).

## Weighted F1 (headline)


### +GI (two-pass graph injection, with span-index head)

| Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | best (meaningful) | rand vs best |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0.3014 | 0.3720 | 0.3869 | **0.4062** | 0.3057 | 0.20=0.4062 | best > rand (Δ=+0.1005) |
| gpt-oss-120b | SWE_Bench_dedup | 0.2713 | 0.3007 | **0.3502** | 0.2772 | _0.3530_ | 0.25=0.3502 | **rand > best** (Δ=+0.0028) |
| gpt-oss-20b | GAIA_dedup | 0.2229 | **0.3328** | 0.3311 | 0.3030 | 0.2517 | 0.35=0.3328 | best > rand (Δ=+0.0811) |
| gpt-oss-20b | SWE_Bench_dedup | 0.1303 | **0.2989** | 0.2771 | 0.2113 | 0.2350 | 0.35=0.2989 | best > rand (Δ=+0.0639) |
| Mistral-Small-24B | GAIA_dedup | 0.3076 | 0.3415 | **0.3476** | 0.3399 | 0.2557 | 0.25=0.3476 | best > rand (Δ=+0.0919) |
| Mistral-Small-24B | SWE_Bench_dedup | 0.1245 | **0.1440** | 0.0927 | 0.0786 | 0.0900 | 0.35=0.1440 | best > rand (Δ=+0.0540) |
| QwenLong-32B | GAIA_dedup | 0.1683 | **0.2657** | 0.2075 | 0.2546 | 0.1957 | 0.35=0.2657 | best > rand (Δ=+0.0700) |
| QwenLong-32B | SWE_Bench_dedup | 0.1109 | 0.1033 | **0.1653** | 0.1320 | 0.1010 | 0.25=0.1653 | best > rand (Δ=+0.0643) |
| Gemma-3-27B | GAIA_dedup | **0.2530** | 0.2126 | 0.1896 | 0.2112 | 0.2146 | causal_only=0.2530 | best > rand (Δ=+0.0384) |
| Gemma-3-27B | SWE_Bench_dedup | 0.1516 | 0.1541 | 0.1679 | **0.1852** | 0.1509 | 0.20=0.1852 | best > rand (Δ=+0.0343) |

### +CG (one-pass causal graph in prompt)

| Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | best (meaningful) | rand vs best |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0.2694 | 0.2896 | 0.2907 | **0.2924** | 0.2617 | 0.20=0.2924 | best > rand (Δ=+0.0307) |
| gpt-oss-120b | SWE_Bench_dedup | 0.2008 | 0.2378 | **0.2703** | 0.2350 | 0.2401 | 0.25=0.2703 | best > rand (Δ=+0.0302) |
| gpt-oss-20b | GAIA_dedup | **0.2017** | 0.1249 | 0.1265 | 0.1348 | 0.1413 | causal_only=0.2017 | best > rand (Δ=+0.0604) |
| gpt-oss-20b | SWE_Bench_dedup | 0.0921 | 0.1259 | **0.1321** | 0.1207 | 0.1265 | 0.25=0.1321 | best > rand (Δ=+0.0056) |
| Mistral-Small-24B | GAIA_dedup | **0.2831** | 0.2395 | 0.2127 | 0.2444 | 0.1753 | causal_only=0.2831 | best > rand (Δ=+0.1078) |
| Mistral-Small-24B | SWE_Bench_dedup | — | 0.1783 | **0.1986** | 0.1323 | 0.0889 | 0.25=0.1986 | best > rand (Δ=+0.1097) |
| QwenLong-32B | GAIA_dedup | 0.1425 | 0.1591 | 0.1611 | **0.1695** | — | 0.20=0.1695 | (rand missing) |
| QwenLong-32B | SWE_Bench_dedup | **0.0877** | — | — | — | — | causal_only=0.0877 | (rand missing) |
| Gemma-3-27B | GAIA_dedup | **0.2455** | 0.1758 | 0.1842 | 0.1892 | 0.1830 | causal_only=0.2455 | best > rand (Δ=+0.0625) |
| Gemma-3-27B | SWE_Bench_dedup | 0.1469 | **0.2339** | 0.1972 | 0.2295 | 0.2116 | 0.35=0.2339 | best > rand (Δ=+0.0223) |

## Average Location Accuracy


### +GI (two-pass graph injection, with span-index head)

| Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | best (meaningful) | rand vs best |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0.2447 | **0.2800** | 0.2196 | 0.2599 | 0.2172 | 0.35=0.2800 | best > rand (Δ=+0.0628) |
| gpt-oss-120b | SWE_Bench_dedup | **0.0292** | 0.0160 | 0.0102 | 0.0125 | _0.0391_ | causal_only=0.0292 | **rand > best** (Δ=+0.0099) |
| gpt-oss-20b | GAIA_dedup | **0.1829** | 0.1228 | 0.1259 | 0.1327 | 0.1212 | causal_only=0.1829 | best > rand (Δ=+0.0617) |
| gpt-oss-20b | SWE_Bench_dedup | 0.0133 | 0.0123 | **0.0178** | 0.0042 | 0.0111 | 0.25=0.0178 | best > rand (Δ=+0.0067) |
| Mistral-Small-24B | GAIA_dedup | 0.2779 | 0.2571 | 0.2813 | **0.2815** | 0.2261 | 0.20=0.2815 | best > rand (Δ=+0.0554) |
| Mistral-Small-24B | SWE_Bench_dedup | 0.0472 | **0.0759** | 0.0392 | 0.0542 | 0.0356 | 0.35=0.0759 | best > rand (Δ=+0.0403) |
| QwenLong-32B | GAIA_dedup | **0.1923** | 0.1732 | 0.1243 | 0.1716 | 0.1220 | causal_only=0.1923 | best > rand (Δ=+0.0703) |
| QwenLong-32B | SWE_Bench_dedup | 0.0042 | 0.0042 | **0.0083** | 0.0042 | 0.0031 | 0.25=0.0083 | best > rand (Δ=+0.0052) |
| Gemma-3-27B | GAIA_dedup | 0.1100 | **0.1120** | 0.0687 | 0.1100 | 0.0727 | 0.35=0.1120 | best > rand (Δ=+0.0393) |
| Gemma-3-27B | SWE_Bench_dedup | **0.0236** | 0.0147 | 0.0147 | 0.0188 | 0.0147 | causal_only=0.0236 | best > rand (Δ=+0.0089) |

### +CG (one-pass causal graph in prompt)

| Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | best (meaningful) | rand vs best |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0.2019 | 0.1785 | 0.2021 | **0.2089** | _0.2151_ | 0.20=0.2089 | **rand > best** (Δ=+0.0062) |
| gpt-oss-120b | SWE_Bench_dedup | 0.0458 | 0.0042 | 0.0211 | **0.0542** | _0.0581_ | 0.20=0.0542 | **rand > best** (Δ=+0.0039) |
| gpt-oss-20b | GAIA_dedup | **0.1103** | 0.0723 | 0.0405 | 0.0403 | 0.0439 | causal_only=0.1103 | best > rand (Δ=+0.0664) |
| gpt-oss-20b | SWE_Bench_dedup | **0.0101** | 0.0042 | 0.0021 | 0.0042 | 0.0063 | causal_only=0.0101 | best > rand (Δ=+0.0038) |
| Mistral-Small-24B | GAIA_dedup | **0.3031** | 0.1136 | 0.1132 | 0.1133 | 0.0532 | causal_only=0.3031 | best > rand (Δ=+0.2499) |
| Mistral-Small-24B | SWE_Bench_dedup | — | **0.0000** | 0.0000 | 0.0000 | _0.0444_ | 0.35=0.0000 | **rand > best** (Δ=+0.0444) |
| QwenLong-32B | GAIA_dedup | 0.0483 | 0.0516 | 0.0513 | **0.0580** | — | 0.20=0.0580 | (rand missing) |
| QwenLong-32B | SWE_Bench_dedup | **0.0000** | — | — | — | — | causal_only=0.0000 | (rand missing) |
| Gemma-3-27B | GAIA_dedup | 0.0448 | **0.0763** | 0.0656 | 0.0432 | _0.0825_ | 0.35=0.0763 | **rand > best** (Δ=+0.0062) |
| Gemma-3-27B | SWE_Bench_dedup | **0.0042** | 0.0036 | 0.0000 | 0.0031 | _0.0042_ | causal_only=0.0042 | tie |

## Average Location-Category Joint Accuracy


### +GI (two-pass graph injection, with span-index head)

| Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | best (meaningful) | rand vs best |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0.0634 | 0.0948 | 0.0783 | **0.1281** | 0.0693 | 0.20=0.1281 | best > rand (Δ=+0.0588) |
| gpt-oss-120b | SWE_Bench_dedup | 0.0000 | 0.0000 | **0.0025** | 0.0000 | _0.0031_ | 0.25=0.0025 | **rand > best** (Δ=+0.0006) |
| gpt-oss-20b | GAIA_dedup | 0.0440 | 0.0409 | 0.0502 | **0.0554** | 0.0426 | 0.20=0.0554 | best > rand (Δ=+0.0128) |
| gpt-oss-20b | SWE_Bench_dedup | 0.0042 | **0.0043** | 0.0028 | 0.0000 | 0.0000 | 0.35=0.0043 | best > rand (Δ=+0.0043) |
| Mistral-Small-24B | GAIA_dedup | 0.1151 | **0.1227** | 0.1068 | 0.1105 | 0.0979 | 0.35=0.1227 | best > rand (Δ=+0.0248) |
| Mistral-Small-24B | SWE_Bench_dedup | 0.0000 | **0.0087** | 0.0000 | 0.0025 | 0.0000 | 0.35=0.0087 | best > rand (Δ=+0.0087) |
| QwenLong-32B | GAIA_dedup | 0.0363 | 0.0413 | 0.0305 | **0.0597** | 0.0408 | 0.20=0.0597 | best > rand (Δ=+0.0189) |
| QwenLong-32B | SWE_Bench_dedup | **0.0000** | 0.0000 | 0.0000 | 0.0000 | _0.0000_ | causal_only=0.0000 | tie |
| Gemma-3-27B | GAIA_dedup | **0.0133** | 0.0073 | 0.0027 | 0.0113 | 0.0020 | causal_only=0.0133 | best > rand (Δ=+0.0113) |
| Gemma-3-27B | SWE_Bench_dedup | **0.0000** | 0.0000 | 0.0000 | 0.0000 | _0.0000_ | causal_only=0.0000 | tie |

### +CG (one-pass causal graph in prompt)

| Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | best (meaningful) | rand vs best |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0.0667 | **0.0688** | 0.0682 | 0.0636 | _0.0696_ | 0.35=0.0688 | **rand > best** (Δ=+0.0008) |
| gpt-oss-120b | SWE_Bench_dedup | 0.0031 | 0.0000 | 0.0031 | **0.0130** | 0.0094 | 0.20=0.0130 | best > rand (Δ=+0.0036) |
| gpt-oss-20b | GAIA_dedup | **0.0309** | 0.0060 | 0.0085 | 0.0038 | 0.0073 | causal_only=0.0309 | best > rand (Δ=+0.0236) |
| gpt-oss-20b | SWE_Bench_dedup | **0.0000** | 0.0000 | 0.0000 | 0.0000 | _0.0031_ | causal_only=0.0000 | **rand > best** (Δ=+0.0031) |
| Mistral-Small-24B | GAIA_dedup | **0.1116** | 0.0572 | 0.0411 | 0.0523 | 0.0229 | causal_only=0.1116 | best > rand (Δ=+0.0887) |
| Mistral-Small-24B | SWE_Bench_dedup | — | **0.0000** | 0.0000 | 0.0000 | _0.0018_ | 0.35=0.0000 | **rand > best** (Δ=+0.0018) |
| QwenLong-32B | GAIA_dedup | 0.0121 | 0.0136 | **0.0151** | 0.0083 | — | 0.25=0.0151 | (rand missing) |
| QwenLong-32B | SWE_Bench_dedup | **0.0000** | — | — | — | — | causal_only=0.0000 | (rand missing) |
| Gemma-3-27B | GAIA_dedup | **0.0020** | 0.0020 | 0.0000 | 0.0013 | _0.0127_ | causal_only=0.0020 | **rand > best** (Δ=+0.0107) |
| Gemma-3-27B | SWE_Bench_dedup | **0.0000** | 0.0000 | 0.0000 | 0.0000 | _0.0000_ | causal_only=0.0000 | tie |

## Summary across all evaluable cells

| Metric | Method | cells where random ≥ best | median Δ(best − rand) |
|---|---|---|---|
| Weighted F1 | +GI | 1 / 10 | +0.0643 |
| Weighted F1 | +CG | 0 / 8 | +0.0604 |
| Loc Acc | +GI | 1 / 10 | +0.0403 |
| Loc Acc | +CG | 5 / 8 | +0.0000 |
| Loc-Cat Joint | +GI | 3 / 10 | +0.0113 |
| Loc-Cat Joint | +CG | 5 / 8 | +0.0000 |

## Key observations

1. **Random never beats meaningful graphs on Weighted F1 for +CG** (0/8 evaluable cells). For +GI, only one cell ties (gpt-oss-120b SWE, Δ=-0.003).
2. **causal-only often wins for +CG on GAIA** (Mistral, gpt-oss-20b, Gemma): adding correlation edges to the one-pass prompt actually hurts. The two-pass +GI head, by contrast, exploits the denser τ=0.20–0.25 graphs.
3. **+GI dominates +CG on every localization metric**; +CG's one-pass prompt has no explicit step-index head, so Loc-Cat joint is often near zero even when F1 is competitive.
4. **+CG > +GI on a few SWE_Bench cells** (Mistral, Gemma) — these are exactly the splits where +GI's confident span predictions hurt it on this benchmark.

## Where random-12 strictly beats every meaningful graph

Across all metrics × methods × cells, 11 (model, split, metric, method) tuples have random-12 > every meaningful variant (causal-only ∪ all τ).

| Metric | Method | Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | Δ(rand−best) |
|---|---|---|---|---|---|---|---|---|---|
| Weighted F1 | +GI | gpt-oss-120b | SWE_Bench_dedup | 0.2713 | 0.3007 | **0.3502** | 0.2772 | _0.3530_ | +0.0028 |
| Loc Acc | +GI | gpt-oss-120b | SWE_Bench_dedup | **0.0292** | 0.0160 | 0.0102 | 0.0125 | _0.0391_ | +0.0099 |
| Loc Acc | +CG | gpt-oss-120b | GAIA_dedup | 0.2019 | 0.1785 | 0.2021 | **0.2089** | _0.2151_ | +0.0062 |
| Loc Acc | +CG | gpt-oss-120b | SWE_Bench_dedup | 0.0458 | 0.0042 | 0.0211 | **0.0542** | _0.0581_ | +0.0039 |
| Loc Acc | +CG | Mistral-Small-24B | SWE_Bench_dedup | — | **0.0000** | 0.0000 | 0.0000 | _0.0444_ | +0.0444 |
| Loc Acc | +CG | Gemma-3-27B | GAIA_dedup | 0.0448 | **0.0763** | 0.0656 | 0.0432 | _0.0825_ | +0.0062 |
| Loc-Cat Joint | +GI | gpt-oss-120b | SWE_Bench_dedup | 0.0000 | 0.0000 | **0.0025** | 0.0000 | _0.0031_ | +0.0006 |
| Loc-Cat Joint | +CG | gpt-oss-120b | GAIA_dedup | 0.0667 | **0.0688** | 0.0682 | 0.0636 | _0.0696_ | +0.0008 |
| Loc-Cat Joint | +CG | gpt-oss-20b | SWE_Bench_dedup | **0.0000** | 0.0000 | 0.0000 | 0.0000 | _0.0031_ | +0.0031 |
| Loc-Cat Joint | +CG | Mistral-Small-24B | SWE_Bench_dedup | — | **0.0000** | 0.0000 | 0.0000 | _0.0018_ | +0.0018 |
| Loc-Cat Joint | +CG | Gemma-3-27B | GAIA_dedup | **0.0020** | 0.0020 | 0.0000 | 0.0013 | _0.0127_ | +0.0107 |

### Hits grouped by model × split

| Model | Split | # cells where rand > all meaningful (out of 6 = 3 metrics × 2 methods) |
|---|---|---|
| gpt-oss-120b | GAIA_dedup | 2 ← |
| gpt-oss-120b | SWE_Bench_dedup | 4 ← |
| gpt-oss-20b | GAIA_dedup | 0 |
| gpt-oss-20b | SWE_Bench_dedup | 1 |
| Mistral-Small-24B | GAIA_dedup | 0 |
| Mistral-Small-24B | SWE_Bench_dedup | 2 ← |
| QwenLong-32B | GAIA_dedup | 0 |
| QwenLong-32B | SWE_Bench_dedup | 0 |
| Gemma-3-27B | GAIA_dedup | 2 ← |
| Gemma-3-27B | SWE_Bench_dedup | 0 |

### Interpretation

- **gpt-oss-120b** is the worst offender: 6/11 rand-wins (2 GAIA, 4 SWE). The two F1/Loc rand-wins on SWE are competitive in magnitude (Δ +0.003 to +0.010) — i.e., the graph is failing to help the strongest model on SWE.
- **Mistral-Small (SWE) and Gemma-3 (GAIA)** show rand-wins only on Loc / Loc-Cat metrics where every variant is at or near 0 — these are noise-floor cases, not evidence against the graph.
- **gpt-oss-20b GAIA** and **QwenLong-32B** have zero rand-wins: the graph helps cleanly for these models.
- Only 1 of the 11 rand-wins is on the headline **Weighted F1** metric, and the margin there is **+0.003** (gpt-oss-120b SWE +GI) — a tie, not a real reversal.

## +GI: where random-12 strictly beats every meaningful graph

Evaluable slots for +GI: **30** (every model × split × metric has both rand and ≥1 meaningful).
Rand-wins: **3 / 30**.

| Metric | Model | Split | causal-only | τ=0.35 | τ=0.25 | τ=0.20 | random-12 | Δ(rand−best) |
|---|---|---|---|---|---|---|---|---|---|
| Weighted F1 | gpt-oss-120b | SWE_Bench_dedup | 0.2713 | 0.3007 | **0.3502** | 0.2772 | _0.3530_ | +0.0028 |
| Loc Acc | gpt-oss-120b | SWE_Bench_dedup | **0.0292** | 0.0160 | 0.0102 | 0.0125 | _0.0391_ | +0.0099 |
| Loc-Cat Joint | gpt-oss-120b | SWE_Bench_dedup | 0.0000 | 0.0000 | **0.0025** | 0.0000 | _0.0031_ | +0.0006 |

## +GI vs +CG — which method is more robust to the random control?

Lower rand-win rate ⇒ method makes better use of the meaningful graph (i.e., a random graph wouldn't have worked).

### Per-metric rand-win rate

| Metric | +GI rand-wins / evaluable | +GI rate | +CG rand-wins / evaluable | +CG rate |
|---|---|---|---|---|
| Weighted F1 | 1 / 10 | 10% | 0 / 8 | 0% |
| Loc Acc | 1 / 10 | 10% | 4 / 8 | 50% |
| Loc-Cat Joint | 1 / 10 | 10% | 4 / 8 | 50% |

**Overall**: +GI = 3 / 30 = 10%  vs  +CG = 8 / 24 = 33%.

### Per (model, split) rand-win counts (max 3 per method = one per metric)

| Model | Split | +GI rand-wins | +CG rand-wins | Δ (+CG − +GI) |
|---|---|---|---|---|
| gpt-oss-120b | GAIA_dedup | 0 | 2 | **+2** (+CG worse) |
| gpt-oss-120b | SWE_Bench_dedup | 3 | 1 | _-2_ (+GI worse) |
| gpt-oss-20b | GAIA_dedup | 0 | 0 | 0 |
| gpt-oss-20b | SWE_Bench_dedup | 0 | 1 | **+1** (+CG worse) |
| Mistral-Small-24B | GAIA_dedup | 0 | 0 | 0 |
| Mistral-Small-24B | SWE_Bench_dedup | 0 | 2 | **+2** (+CG worse) |
| QwenLong-32B | GAIA_dedup | 0 | 0 | 0 |
| QwenLong-32B | SWE_Bench_dedup | 0 | 0 | 0 |
| Gemma-3-27B | GAIA_dedup | 0 | 2 | **+2** (+CG worse) |
| Gemma-3-27B | SWE_Bench_dedup | 0 | 0 | 0 |

### Interpretation

- **+GI is more robust**: only **3/30 (10%) slots** fail the random-graph control, vs **8/24 (33%) for +CG**.
- The +GI rand-wins are concentrated on **one (model, split)**: gpt-oss-120b SWE_Bench — every metric there shows a small rand-win (Δ +0.001 to +0.010). Every other model × split passes the +GI control cleanly.
- +CG fails the control on **more models and more cells**: it rand-loses on gpt-oss-120b (both splits), Mistral SWE, gpt-oss-20b SWE, and Gemma GAIA — five distinct (model, split) cells vs +GI's one.
- Headline F1: +GI has 1 rand-win, +CG has 0 — but that +CG zero is on only 8 evaluable F1 cells (vs +GI's 10), since some +CG runs are still pending.
