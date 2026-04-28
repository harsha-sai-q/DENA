# DENA-Bench v0 Report

## Main result

Best cost-adjusted system in this toy benchmark: **DENA_top1**.

DENA_top2 achieved quality score **0.974** with average estimated cost **0.337**. AllExperts_proxy achieved quality score **0.977** with average estimated cost **0.788**. That is an estimated cost reduction of **57.2%** for DENA_top2 compared with AllExperts_proxy.

Compared with GeneralOnly_proxy, DENA_top2 improved route F1 from **0.000** to **0.741**.

## Summary table

| system | route_f1 | route_recall | answer_score | quality_score | cost_adjusted_score | avg_estimated_cost | avg_parallel_latency | avg_activation_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DENA_top1 | 0.903 | 0.975 | 0.939 | 0.961 | 0.932 | 0.194 | 0.194 | 1.576 |
| DENA_top2 | 0.741 | 1.000 | 0.949 | 0.974 | 0.928 | 0.337 | 0.243 | 3.152 |
| DENA_top3 | 0.602 | 1.000 | 0.949 | 0.974 | 0.911 | 0.485 | 0.300 | 4.727 |
| OracleRouter_upper_bound | 0.733 | 0.960 | 0.929 | 0.950 | 0.908 | 0.309 | 0.227 | 3.152 |
| AllExperts_proxy | 0.420 | 1.000 | 0.949 | 0.977 | 0.883 | 0.788 | 0.315 | 7.879 |
| RandomRouter_proxy | 0.432 | 0.636 | 0.803 | 0.741 | 0.701 | 0.299 | 0.213 | 3.152 |
| GeneralOnly_proxy | 0.000 | 0.000 | 0.449 | 0.302 | 0.293 | 0.063 | 0.063 | 1.576 |

## Category table

| system | category | route_f1 | answer_score | quality_score | cost_adjusted_score | avg_estimated_cost |
| --- | --- | --- | --- | --- | --- | --- |
| AllExperts_proxy | architecture_research | 0.393 | 0.833 | 0.925 | 0.842 | 0.688 |
| AllExperts_proxy | code_debug | 0.333 | 1.000 | 1.000 | 0.920 | 0.667 |
| AllExperts_proxy | definition | 0.413 | 0.944 | 0.975 | 0.855 | 1.000 |
| AllExperts_proxy | math_exact | 0.333 | 1.000 | 1.000 | 0.940 | 0.500 |
| AllExperts_proxy | mixed_domain | 0.714 | 1.000 | 1.000 | 0.844 | 1.300 |
| DENA_top1 | architecture_research | 0.875 | 0.792 | 0.878 | 0.845 | 0.223 |
| DENA_top1 | code_debug | 0.889 | 1.000 | 1.000 | 0.975 | 0.170 |
| DENA_top1 | definition | 0.778 | 0.944 | 0.975 | 0.936 | 0.260 |
| DENA_top1 | math_exact | 1.000 | 1.000 | 1.000 | 0.993 | 0.050 |
| DENA_top1 | mixed_domain | 0.960 | 1.000 | 0.970 | 0.921 | 0.326 |
| DENA_top2 | architecture_research | 0.729 | 0.833 | 0.925 | 0.876 | 0.356 |
| DENA_top2 | code_debug | 0.639 | 1.000 | 0.983 | 0.949 | 0.253 |
| DENA_top2 | definition | 0.778 | 0.944 | 0.975 | 0.903 | 0.520 |
| DENA_top2 | math_exact | 0.667 | 1.000 | 1.000 | 0.986 | 0.110 |
| DENA_top2 | mixed_domain | 0.960 | 1.000 | 1.000 | 0.926 | 0.552 |
| DENA_top3 | architecture_research | 0.575 | 0.833 | 0.925 | 0.869 | 0.426 |
| DENA_top3 | code_debug | 0.500 | 1.000 | 0.983 | 0.939 | 0.343 |
| DENA_top3 | definition | 0.600 | 0.944 | 0.975 | 0.893 | 0.620 |
| DENA_top3 | math_exact | 0.500 | 1.000 | 1.000 | 0.959 | 0.310 |
| DENA_top3 | mixed_domain | 0.931 | 1.000 | 1.000 | 0.888 | 0.866 |
| GeneralOnly_proxy | architecture_research | 0.000 | 0.708 | 0.419 | 0.411 | 0.055 |
| GeneralOnly_proxy | code_debug | 0.000 | 0.667 | 0.400 | 0.392 | 0.053 |
| GeneralOnly_proxy | definition | 0.000 | 0.472 | 0.312 | 0.300 | 0.080 |
| GeneralOnly_proxy | math_exact | 0.000 | 0.037 | 0.117 | 0.111 | 0.040 |
| GeneralOnly_proxy | mixed_domain | 0.000 | 0.403 | 0.282 | 0.266 | 0.104 |
| OracleRouter_upper_bound | architecture_research | 0.750 | 0.750 | 0.887 | 0.839 | 0.348 |
| OracleRouter_upper_bound | code_debug | 0.667 | 1.000 | 1.000 | 0.963 | 0.267 |
| OracleRouter_upper_bound | definition | 0.778 | 0.944 | 0.975 | 0.932 | 0.320 |
| OracleRouter_upper_bound | math_exact | 0.667 | 1.000 | 1.000 | 0.973 | 0.200 |
| OracleRouter_upper_bound | mixed_domain | 0.840 | 1.000 | 0.880 | 0.818 | 0.462 |
| RandomRouter_proxy | architecture_research | 0.500 | 0.833 | 0.812 | 0.770 | 0.312 |
| RandomRouter_proxy | code_debug | 0.222 | 0.833 | 0.592 | 0.564 | 0.208 |
| RandomRouter_proxy | definition | 0.494 | 0.944 | 0.900 | 0.847 | 0.390 |
| RandomRouter_proxy | math_exact | 0.417 | 0.625 | 0.662 | 0.634 | 0.205 |
| RandomRouter_proxy | mixed_domain | 0.522 | 0.830 | 0.743 | 0.686 | 0.426 |

## Correct interpretation

DENA-Bench v0 supports the claim that DENA can be implemented, benchmarked, and compared using routing, answer quality, and efficiency metrics in a controlled toy setting. It does not yet prove that DENA outperforms real dense transformers or true MoE models.

## Next benchmark upgrades

1. Replace rule-based experts with neural experts.
2. Add a trainable router.
3. Use public task datasets.
4. Add wall-clock latency and token/FLOP cost.
5. Add ablation tests for memory, verifier, aggregation, and feedback.
6. Compare against real dense, MoE-like, ensemble, and RAG baselines.
