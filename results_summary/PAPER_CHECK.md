# Paper check

Every value printed in the paper that this package recomputes, compared at the
paper's printed precision with the value produced by your run of the scripts.

**774 match, 0 mismatch, 9 approximate, 0 not computed** (not computed: the inputs have not been produced).

Exact agreement is expected when the scores are computed from the paper's generations; outputs you
regenerate yourself differ in wording (GPU non-determinism) and give close, not identical, values.

| Where | Check | Paper | Your run | Result |
|---|---|---|---|---|
| Table 1 | `T1.biomistral-7b.efs` | 0.493 | 0.493212 | match |
| Table 1 | `T1.biomistral-7b.efs_std` | 0.002 | 0.00231955 | match |
| Table 1 | `T1.biomistral-7b.dist_per_gen` | 10.423 | 10.4232 | match |
| Table 1 | `T1.biomistral-7b.escalation_pct` | 7.6 | 7.56667 | match |
| Table 1 | `T1.biomistral-7b.attenuation_pct` | 0.3 | 0.348352 | match |
| Table 1 | `T1.biomistral-7b.omission_pct` | 82.9 | 82.9422 | match |
| Table 1 | `T1.biomistral-7b.fabrication_pct` | 3.2 | 3.17515 | match |
| Table 1 | `T1.biomistral-7b.scope_shift_pct` | 3.3 | 3.30649 | match |
| Table 1 | `T1.biomistral-7b.severity_warp_pct` | 2.7 | 2.66118 | match |
| Table 1 | `T1.llama-3.1-8b.efs` | 0.481 | 0.480546 | match |
| Table 1 | `T1.llama-3.1-8b.efs_std` | 0.005 | 0.00481439 | match |
| Table 1 | `T1.llama-3.1-8b.dist_per_gen` | 10.611 | 10.6107 | match |
| Table 1 | `T1.llama-3.1-8b.escalation_pct` | 10.6 | 10.6081 | match |
| Table 1 | `T1.llama-3.1-8b.attenuation_pct` | 1.5 | 1.53708 | match |
| Table 1 | `T1.llama-3.1-8b.omission_pct` | 72.3 | 72.3325 | match |
| Table 1 | `T1.llama-3.1-8b.fabrication_pct` | 4.9 | 4.931 | match |
| Table 1 | `T1.llama-3.1-8b.scope_shift_pct` | 6.4 | 6.44003 | match |
| Table 1 | `T1.llama-3.1-8b.severity_warp_pct` | 4.2 | 4.15124 | match |
| Table 1 | `T1.openbio-70b.efs` | 0.478 | 0.478163 | match |
| Table 1 | `T1.openbio-70b.efs_std` | 0.001 | 0.00140527 | match |
| Table 1 | `T1.openbio-70b.dist_per_gen` | 10.660 | 10.6601 | match |
| Table 1 | `T1.openbio-70b.escalation_pct` | 10.4 | 10.4361 | match |
| Table 1 | `T1.openbio-70b.attenuation_pct` | 1.2 | 1.22285 | match |
| Table 1 | `T1.openbio-70b.omission_pct` | 75.3 | 75.2806 | match |
| Table 1 | `T1.openbio-70b.fabrication_pct` | 4.1 | 4.12083 | match |
| Table 1 | `T1.openbio-70b.scope_shift_pct` | 5.0 | 5.04216 | match |
| Table 1 | `T1.openbio-70b.severity_warp_pct` | 3.9 | 3.89748 | match |
| Table 1 | `T1.mistral-7b.efs` | 0.478 | 0.477945 | match |
| Table 1 | `T1.mistral-7b.efs_std` | 0.002 | 0.00221882 | match |
| Table 1 | `T1.mistral-7b.dist_per_gen` | 10.619 | 10.619 | match |
| Table 1 | `T1.mistral-7b.escalation_pct` | 11.0 | 10.9865 | match |
| Table 1 | `T1.mistral-7b.attenuation_pct` | 1.3 | 1.26121 | match |
| Table 1 | `T1.mistral-7b.omission_pct` | 73.4 | 73.4361 | match |
| Table 1 | `T1.mistral-7b.fabrication_pct` | 4.5 | 4.50673 | match |
| Table 1 | `T1.mistral-7b.scope_shift_pct` | 5.6 | 5.56614 | match |
| Table 1 | `T1.mistral-7b.severity_warp_pct` | 4.2 | 4.24327 | match |
| Table 1 | `T1.med42-8b.efs` | 0.477 | 0.476611 | match |
| Table 1 | `T1.med42-8b.efs_std` | 0.006 | 0.00626369 | match |
| Table 1 | `T1.med42-8b.dist_per_gen` | 10.521 | 10.5208 | match |
| Table 1 | `T1.med42-8b.escalation_pct` | 10.5 | 10.4724 | match |
| Table 1 | `T1.med42-8b.attenuation_pct` | 1.6 | 1.62376 | match |
| Table 1 | `T1.med42-8b.omission_pct` | 69.6 | 69.6464 | match |
| Table 1 | `T1.med42-8b.fabrication_pct` | 7.1 | 7.05516 | match |
| Table 1 | `T1.med42-8b.scope_shift_pct` | 7.2 | 7.157 | match |
| Table 1 | `T1.med42-8b.severity_warp_pct` | 4.0 | 4.04526 | match |
| Table 1 | `T1.llama-3.3-70b.efs` | 0.447 | 0.447241 | match |
| Table 1 | `T1.llama-3.3-70b.efs_std` | 0.008 | 0.00779435 | match |
| Table 1 | `T1.llama-3.3-70b.dist_per_gen` | 11.113 | 11.1131 | match |
| Table 1 | `T1.llama-3.3-70b.escalation_pct` | 11.3 | 11.2748 | match |
| Table 1 | `T1.llama-3.3-70b.attenuation_pct` | 1.9 | 1.93894 | match |
| Table 1 | `T1.llama-3.3-70b.omission_pct` | 67.5 | 67.4826 | match |
| Table 1 | `T1.llama-3.3-70b.fabrication_pct` | 7.7 | 7.68613 | match |
| Table 1 | `T1.llama-3.3-70b.scope_shift_pct` | 7.3 | 7.26299 | match |
| Table 1 | `T1.llama-3.3-70b.severity_warp_pct` | 4.4 | 4.35458 | match |
| Table 1 | `T1.gemma-4-4b.efs` | 0.432 | 0.43218 | match |
| Table 1 | `T1.gemma-4-4b.efs_std` | 0.011 | 0.0114195 | match |
| Table 1 | `T1.gemma-4-4b.dist_per_gen` | 11.278 | 11.278 | match |
| Table 1 | `T1.gemma-4-4b.escalation_pct` | 13.4 | 13.3741 | match |
| Table 1 | `T1.gemma-4-4b.attenuation_pct` | 1.8 | 1.84726 | match |
| Table 1 | `T1.gemma-4-4b.omission_pct` | 58.4 | 58.4367 | match |
| Table 1 | `T1.gemma-4-4b.fabrication_pct` | 12.4 | 12.4136 | match |
| Table 1 | `T1.gemma-4-4b.scope_shift_pct` | 8.7 | 8.69795 | match |
| Table 1 | `T1.gemma-4-4b.severity_warp_pct` | 5.2 | 5.23038 | match |
| Table 1 | `T1.med42-70b.efs` | 0.420 | 0.420133 | match |
| Table 1 | `T1.med42-70b.efs_std` | 0.010 | 0.0103 | match |
| Table 1 | `T1.med42-70b.dist_per_gen` | 11.524 | 11.5238 | match |
| Table 1 | `T1.med42-70b.escalation_pct` | 12.6 | 12.6033 | match |
| Table 1 | `T1.med42-70b.attenuation_pct` | 2.7 | 2.72727 | match |
| Table 1 | `T1.med42-70b.omission_pct` | 60.9 | 60.8574 | match |
| Table 1 | `T1.med42-70b.fabrication_pct` | 10.3 | 10.2893 | match |
| Table 1 | `T1.med42-70b.scope_shift_pct` | 8.0 | 7.97521 | match |
| Table 1 | `T1.med42-70b.severity_warp_pct` | 5.5 | 5.54752 | match |
| Table 1 | `T1.gemma-4-moe.efs` | 0.418 | 0.417546 | match |
| Table 1 | `T1.gemma-4-moe.efs_std` | 0.020 | 0.0204843 | match |
| Table 1 | `T1.gemma-4-moe.dist_per_gen` | 11.573 | 11.5732 | match |
| Table 1 | `T1.gemma-4-moe.escalation_pct` | 13.3 | 13.2953 | match |
| Table 1 | `T1.gemma-4-moe.attenuation_pct` | 2.1 | 2.10873 | match |
| Table 1 | `T1.gemma-4-moe.omission_pct` | 58.2 | 58.2472 | match |
| Table 1 | `T1.gemma-4-moe.fabrication_pct` | 12.3 | 12.2512 | match |
| Table 1 | `T1.gemma-4-moe.scope_shift_pct` | 8.7 | 8.68179 | match |
| Table 1 | `T1.gemma-4-moe.severity_warp_pct` | 5.4 | 5.41583 | match |
| Table 1 | `T1.gemma-4-31b.efs` | 0.416 | 0.416055 | match |
| Table 1 | `T1.gemma-4-31b.efs_std` | 0.013 | 0.0133752 | match |
| Table 1 | `T1.gemma-4-31b.dist_per_gen` | 11.608 | 11.6083 | match |
| Table 1 | `T1.gemma-4-31b.escalation_pct` | 13.4 | 13.4191 | match |
| Table 1 | `T1.gemma-4-31b.attenuation_pct` | 1.8 | 1.84596 | match |
| Table 1 | `T1.gemma-4-31b.omission_pct` | 59.4 | 59.399 | match |
| Table 1 | `T1.gemma-4-31b.fabrication_pct` | 12.0 | 11.9783 | match |
| Table 1 | `T1.gemma-4-31b.scope_shift_pct` | 8.0 | 7.99918 | match |
| Table 1 | `T1.gemma-4-31b.severity_warp_pct` | 5.4 | 5.35842 | match |
| Table 1 | `T1.medgemma-27b.efs` | 0.415 | 0.41464 | match |
| Table 1 | `T1.medgemma-27b.efs_std` | 0.015 | 0.0145146 | match |
| Table 1 | `T1.medgemma-27b.dist_per_gen` | 11.629 | 11.6292 | match |
| Table 1 | `T1.medgemma-27b.escalation_pct` | 14.3 | 14.342 | match |
| Table 1 | `T1.medgemma-27b.attenuation_pct` | 3.1 | 3.08645 | match |
| Table 1 | `T1.medgemma-27b.omission_pct` | 55.0 | 54.9675 | match |
| Table 1 | `T1.medgemma-27b.fabrication_pct` | 12.7 | 12.7143 | match |
| Table 1 | `T1.medgemma-27b.scope_shift_pct` | 8.8 | 8.76798 | match |
| Table 1 | `T1.medgemma-27b.severity_warp_pct` | 6.1 | 6.12172 | match |
| 6.1 | `S6.1.mean_efs_3seeds` | 0.450 | 0.450388 | match |
| 6.1 | `S6.1.efs_range_min` | 0.415 | 0.41464 | match |
| 6.1 | `S6.1.efs_range_max` | 0.493 | 0.493212 | match |
| 6.1 | `S6.1.crossseed_std_min` | 0.001 | 0.00140527 | match |
| 6.1 | `S6.1.crossseed_std_max` | 0.020 | 0.0204843 | match |
| 6.1 | `S6.1.crossseed_std_mean` | 0.009 | 0.00862816 | match |
| 6.1 | `S6.1.med_mean` | 0.457 | 0.456552 | match |
| 6.1 | `S6.1.gen_mean` | 0.445 | 0.445252 | match |
| Table A18 | `A18.omission.count` | 135,412 | 135412 | match |
| Table A18 | `A18.omission.pct` | 66.3 | 66.3069 | match |
| Table A18 | `A18.omission.per_gen` | 7.327 | 7.32749 | match |
| Table A18 | `A18.escalation.count` | 23,979 | 23979 | match |
| Table A18 | `A18.escalation.pct` | 11.7 | 11.7417 | match |
| Table A18 | `A18.escalation.per_gen` | 1.298 | 1.29756 | match |
| Table A18 | `A18.fabrication.count` | 17,205 | 17205 | match |
| Table A18 | `A18.fabrication.pct` | 8.4 | 8.42474 | match |
| Table A18 | `A18.fabrication.per_gen` | 0.931 | 0.931006 | match |
| Table A18 | `A18.scope_shift.count` | 14,397 | 14397 | match |
| Table A18 | `A18.scope_shift.pct` | 7.0 | 7.04975 | match |
| Table A18 | `A18.scope_shift.per_gen` | 0.779 | 0.779058 | match |
| Table A18 | `A18.severity_warp.count` | 9,548 | 9548 | match |
| Table A18 | `A18.severity_warp.pct` | 4.7 | 4.67535 | match |
| Table A18 | `A18.severity_warp.per_gen` | 0.517 | 0.516667 | match |
| Table A18 | `A18.attenuation.count` | 3,679 | 3679 | match |
| Table A18 | `A18.attenuation.pct` | 1.8 | 1.80149 | match |
| Table A18 | `A18.attenuation.per_gen` | 0.199 | 0.19908 | match |
| Table A18 | `A18.total` | 204,220 | 204220 | match |
| Table A18 | `A18.total_per_gen` | 11.051 | 11.0509 | match |
| Table A1 | `A1.model_H` | 409.8 | 409.767 | match |
| Table A1 | `A1.model_p` | < 1e-81 | 7.8452e-82 | match |
| Table A1 | `A1.task_H` | 113.5 | 113.544 | match |
| Table A1 | `A1.task_p` | < 1e-24 | 2.2087e-25 | match |
| Table A1 | `A1.specialty_H` | 966.9 | 966.884 | match |
| Table A1 | `A1.specialty_p` | < 1e-208 | 2.74705e-209 | match |
| Table A1 | `A1.scale_gen_r` | -0.044 | -0.0437876 | match |
| Table A1 | `A1.scale_gen_p` | < 1e-08 | 2.6008e-09 | match |
| Table A1 / 6.1 | `A1.scale_model_r` | -0.281 | -0.281484 | match |
| Table A1 / 6.1 | `A1.scale_model_p` | 0.402 | 0.401728 | match |
| Table A1 | `A1.med_U` | 4.4e+07 | 4.36521e+07 | match |
| Table A1 / 6.1 | `A1.med_p` | 3e-04 | 0.000266699 | match |
| Table A1 / 6.5 | `A1.domain_H` | 135.2 | 135.179 | match |
| 6.5 | `A1.domain_p` | 4.4e-30 | 4.42921e-30 | match |
| App. B | `B.friedman_blocks_n` | 1,680 | 1680 | match |
| App. B | `B.friedman_blocks_chi2` | 1863.6 | 1863.6 | match |
| App. B | `B.friedman_episodes_n` | 560 | 560 | match |
| App. B | `B.friedman_episodes_chi2` | 1437.1 | 1437.09 | match |
| App. B | `B.med_minus_gen` | 0.011 | 0.0112998 | match |
| App. B | `B.med_minus_gen_ci_lo` | 0.008 | 0.00849022 | match |
| App. B | `B.med_minus_gen_ci_hi` | 0.014 | 0.0141645 | match |
| App. B | `B.med_minus_gen_wilcoxon_p` | 1.1e-16 | 1.08186e-16 | match |
| App. B | `B.n_episodes` | 560 | 560 | match |
| App. B | `B.domain_doc_H` | 28.9 | 28.8952 | match |
| App. B | `B.domain_doc_p` | 5.3e-07 | 5.31471e-07 | match |
| App. B | `B.clin_vs_sci_p` | 0.008 | 0.00805739 | match |
| App. B | `B.clin_vs_legal_p` | < 0.002 | 2.53944e-07 | match |
| App. B | `B.sci_vs_legal_p` | < 0.002 | 0.00170582 | match |
| App. B | `B.paired.prompt.llama_med42.p` | >= 0.11 | 0.719917 | match |
| App. B | `B.paired.fewshot.llama_med42.p` | >= 0.11 | 0.377748 | match |
| App. B | `B.paired.posthoc.llama_med42.p` | >= 0.11 | 0.111793 | match |
| App. B | `B.paired.prompt.mistral.p` | >= 0.07 | 0.624632 | match |
| App. B | `B.paired.fewshot.mistral.p` | >= 0.07 | 0.0728437 | match |
| App. B | `B.paired.posthoc.mistral.p` | >= 0.07 | 0.389233 | match |
| App. B | `B.paired.prompt.gemma.p` | < 1e-11 | 2.28091e-21 | match |
| App. B | `B.paired.fewshot.gemma.p` | < 1e-11 | 4.21202e-33 | match |
| App. B | `B.paired.posthoc.gemma.p` | < 1e-11 | 4.94243e-12 | match |
| App. B | `B.paired.combo.three.delta` | 0.009 | 0.00874124 | match |
| App. B | `B.paired.combo.three.p` | 0.001 | 0.00128776 | match |
| App. B | `B.paired.constrained.three.delta` | -0.010 | -0.0101148 | match |
| App. B | `B.paired.constrained.three.p` | 0.02 | 0.0204895 | match |
| Table A5 | `A5.baseline.efs` | 0.477 | 0.476616 | match |
| Table A5 | `A5.baseline.efs_sd` | 0.217 | 0.217034 | match |
| Table A5 | `A5.prompt.efs` | 0.474 | 0.474341 | match |
| Table A5 | `A5.prompt.efs_sd` | 0.216 | 0.21576 | match |
| Table A5 | `A5.prompt.delta` | -0.002 | -0.00227548 | match |
| Table A5 | `A5.prompt.cohen_d` | -0.011 | -0.0105153 | match |
| Table A5 | `A5.prompt.mwu_p` | 0.959 | 0.959015 | match |
| Table A5 | `A5.fewshot.efs` | 0.471 | 0.471095 | match |
| Table A5 | `A5.fewshot.efs_sd` | 0.216 | 0.215842 | match |
| Table A5 | `A5.fewshot.delta` | -0.006 | -0.00552156 | match |
| Table A5 | `A5.fewshot.cohen_d` | -0.026 | -0.025511 | match |
| Table A5 | `A5.fewshot.mwu_p` | 0.719 | 0.719278 | match |
| Table A5 | `A5.posthoc.efs` | 0.480 | 0.480204 | match |
| Table A5 | `A5.posthoc.efs_sd` | 0.214 | 0.214394 | match |
| Table A5 | `A5.posthoc.delta` | 0.004 | 0.00358821 | match |
| Table A5 | `A5.posthoc.cohen_d` | 0.017 | 0.0166338 | match |
| Table A5 | `A5.posthoc.mwu_p` | 0.653 | 0.652929 | match |
| Table A5 | `A5.T0.3.efs` | 0.517 | 0.516747 | match |
| Table A5 | `A5.T0.3.efs_sd` | 0.170 | 0.170375 | match |
| Table A5 | `A5.T0.3.delta` | 0.040 | 0.0401312 | match |
| Table A5 | `A5.T0.3.cohen_d` | 0.189 | 0.189113 | match |
| Table A5 | `A5.T0.3.mwu_p` | 0.041 | 0.0413742 | match |
| Table A5 | `A5.T0.7.efs` | 0.508 | 0.508218 | match |
| Table A5 | `A5.T0.7.efs_sd` | 0.201 | 0.200749 | match |
| Table A5 | `A5.T0.7.delta` | 0.032 | 0.031602 | match |
| Table A5 | `A5.T0.7.cohen_d` | 0.147 | 0.146829 | match |
| Table A5 | `A5.T0.7.mwu_p` | 0.057 | 0.057457 | match |
| Table A6 | `A6.baseline.efs` | 0.478 | 0.4776 | match |
| Table A6 | `A6.baseline.n` | 1,728 | 1728 | match |
| Table A6 | `A6.strict.n` | 1,728 | 1728 | match |
| Table A6 | `A6.strict.efs` | 0.469 | 0.469132 | match |
| Table A6 | `A6.strict.delta` | -0.008 | -0.00846809 | match |
| Table A6 | `A6.strict.mwu_p` | 0.332 | 0.331851 | match |
| Table A6 | `A6.cot.n` | 1,728 | 1728 | match |
| Table A6 | `A6.cot.efs` | 0.478 | 0.478488 | match |
| Table A6 | `A6.cot.delta` | 0.001 | 0.000887634 | match |
| Table A6 | `A6.cot.mwu_p` | 0.631 | 0.630897 | match |
| Table A6 | `A6.combo.n` | 1,728 | 1728 | match |
| Table A6 | `A6.combo.efs` | 0.486 | 0.486341 | match |
| Table A6 | `A6.combo.delta` | 0.009 | 0.00874124 | match |
| Table A6 | `A6.combo.mwu_p` | 0.213 | 0.212679 | match |
| Table A6 | `A6.constrained.n` | 1,728 | 1728 | match |
| Table A6 | `A6.constrained.efs` | 0.467 | 0.467485 | match |
| Table A6 | `A6.constrained.delta` | -0.010 | -0.0101148 | match |
| Table A6 | `A6.constrained.mwu_p` | 0.231 | 0.230569 | match |
| Table A7 | `A7.mistral-7b.baseline.efs` | 0.480 | 0.479568 | match |
| Table A7 | `A7.gemma-4-4b.baseline.efs` | 0.420 | 0.419685 | match |
| Table A7 | `A7.mistral-7b.prompt.efs` | 0.475 | 0.474744 | match |
| Table A7 | `A7.mistral-7b.prompt.delta` | -0.005 | -0.00482369 | match |
| Table A7 | `A7.mistral-7b.prompt.cohen_d` | -0.023 | -0.0225768 | match |
| Table A7 | `A7.mistral-7b.prompt.mwu_p` | 0.780 | 0.77975 | match |
| Table A7 | `A7.mistral-7b.fewshot.efs` | 0.488 | 0.487626 | match |
| Table A7 | `A7.mistral-7b.fewshot.delta` | 0.008 | 0.00805748 | match |
| Table A7 | `A7.mistral-7b.fewshot.cohen_d` | 0.038 | 0.0382546 | match |
| Table A7 | `A7.mistral-7b.fewshot.mwu_p` | 0.491 | 0.49072 | match |
| Table A7 | `A7.mistral-7b.posthoc.efs` | 0.484 | 0.483527 | match |
| Table A7 | `A7.mistral-7b.posthoc.delta` | 0.004 | 0.0039587 | match |
| Table A7 | `A7.mistral-7b.posthoc.cohen_d` | 0.019 | 0.0187785 | match |
| Table A7 | `A7.mistral-7b.posthoc.mwu_p` | 0.862 | 0.862343 | match |
| Table A7 | `A7.gemma-4-4b.prompt.efs` | 0.480 | 0.480204 | match |
| Table A7 | `A7.gemma-4-4b.prompt.delta` | 0.061 | 0.0605196 | match |
| Table A7 | `A7.gemma-4-4b.prompt.cohen_d` | 0.278 | 0.277991 | match |
| Table A7 | `A7.gemma-4-4b.fewshot.efs` | 0.501 | 0.501343 | match |
| Table A7 | `A7.gemma-4-4b.fewshot.delta` | 0.082 | 0.0816579 | match |
| Table A7 | `A7.gemma-4-4b.fewshot.cohen_d` | 0.379 | 0.378633 | match |
| Table A7 | `A7.gemma-4-4b.posthoc.efs` | 0.461 | 0.461418 | match |
| Table A7 | `A7.gemma-4-4b.posthoc.delta` | 0.042 | 0.0417329 | match |
| Table A7 | `A7.gemma-4-4b.posthoc.cohen_d` | 0.190 | 0.189504 | match |
| Table A7 | `A7.gemma-4-4b.posthoc.mwu_p` | 0.002 | 0.0024344 | match |
| Table A7 | `A7.gemma-4-4b.prompt.mwu_p` | < 1e-05 | 3.03862e-06 | match |
| Table A7 | `A7.gemma-4-4b.fewshot.mwu_p` | < 1e-08 | 9.79893e-10 | match |
| 6.5 | `X.clinical3.efs` | 0.478 | 0.4776 | match |
| 6.5 | `X.legal.efs` | 0.359 | 0.359091 | match |
| 6.5 | `X.scientific.efs` | 0.427 | 0.426716 | match |
| Fig. A1 | `X.clinical.efs` | 0.443 | 0.442616 | match |
| Fig. A1 | `X.clinical.efs_sd` | 0.220 | 0.220068 | match |
| Fig. A1 | `X.legal.efs_sd` | 0.337 | 0.336592 | match |
| Fig. A1 | `X.scientific.efs_sd` | 0.257 | 0.257355 | match |
| J.3 | `X.clinical.pct_floor` | 9.0 | 8.96465 | match |
| J.3 | `X.scientific.pct_floor` | 12.8 | 12.8472 | match |
| J.3 | `X.legal.pct_floor` | 34.7 | 34.7222 | match |
| J.3 | `X.clinical.pct_clamp_binds` | 4.8 | 4.82955 | match |
| J.3 | `X.scientific.pct_clamp_binds` | 5.2 | 5.20833 | match |
| J.3 | `X.legal.pct_clamp_binds` | 20.7 | 20.7465 | match |
| J.3 | `X.clinical.unclamped` | 0.422 | 0.421924 | match |
| J.3 | `X.scientific.unclamped` | 0.411 | 0.411233 | match |
| J.3 | `X.legal.unclamped` | 0.209 | 0.208665 | match |
| J.3 | `X.clinical.rank_rho_unclamped` | 0.964 | 0.963636 | match |
| J.3 | `X.legal.rank_rho_unclamped` | 1.000 | 1 | match |
| J.3 | `X.scientific.rank_rho_unclamped` | 1.000 | 1 | match |
| App. E | `X.clinical.no_omission` | 0.747 | 0.74718 | match |
| App. E | `X.legal.no_omission` | 0.574 | 0.573779 | match |
| App. E | `X.scientific.no_omission` | 0.620 | 0.620386 | match |
| App. E | `X.clinical.any_escalation_pct` | 57 | 56.8182 | match |
| App. E | `X.legal.any_escalation_pct` | 38 | 37.5868 | match |
| App. E | `X.scientific.any_escalation_pct` | 70 | 69.5312 | match |
| App. Q | `X.legal.core4` | 0.392 | 0.392363 | match |
| App. Q | `X.clinical3.core4` | 0.541 | 0.54146 | match |
| App. Q | `X.clinical.core4` | 0.510 | 0.510046 | match |
| F.1 | `X.clinical.efs_w` | 0.531 | 0.531266 | match |
| F.1 | `X.legal.efs_w` | 0.378 | 0.378054 | match |
| F.1 | `X.scientific.efs_w` | 0.411 | 0.411147 | match |
| F.1 | `X.clinical.r_efs_efsw` | 0.874 | 0.874202 | match |
| F.1 | `X.legal.r_efs_efsw` | 0.893 | 0.89307 | match |
| F.1 | `X.clinical.rank_rho_efsw` | 0.982 | 0.981818 | match |
| F.1 | `X.legal.rank_rho_efsw` | 1.000 | 1 | match |
| F.1 | `X.scientific.rank_rho_efsw` | 1.000 | 1 | match |
| F.1 | `F1.pooled_n` | 8,640 | 8640 | match |
| F.1 | `F1.pooled_r` | 0.875 | 0.874958 | match |
| F.1 | `F1.per_model_r` | 0.992 | 0.991905 | match |
| Table 2 | `T2.n_scored` | 7,005 | 7005 | match |
| Table 2 | `T2.Phi-4.legal` | 0.416 | 0.415999 | match |
| Table 2 | `T2.Phi-4.legal.sd` | 0.045 | 0.0449256 | match |
| Table 2 | `T2.Phi-4.medical` | 0.406 | 0.40596 | match |
| Table 2 | `T2.Phi-4.medical.sd` | 0.057 | 0.0565006 | match |
| Table 2 | `T2.Phi-4.scientific` | 0.483 | 0.48259 | match |
| Table 2 | `T2.Phi-4.scientific.sd` | 0.006 | 0.00579418 | match |
| Table 2 | `T2.gpt-4o.legal` | 0.342 | 0.341819 | match |
| Table 2 | `T2.gpt-4o.legal.sd` | 0.037 | 0.0369061 | match |
| Table 2 | `T2.gpt-4o.medical` | 0.373 | 0.373352 | match |
| Table 2 | `T2.gpt-4o.medical.sd` | 0.045 | 0.0452303 | match |
| Table 2 | `T2.gpt-4o.scientific` | 0.468 | 0.467787 | match |
| Table 2 | `T2.gpt-4o.scientific.sd` | 0.016 | 0.016011 | match |
| Table 2 | `T2.Mistral-Large-3.legal` | 0.302 | 0.302351 | match |
| Table 2 | `T2.Mistral-Large-3.legal.sd` | 0.034 | 0.0335775 | match |
| Table 2 | `T2.Mistral-Large-3.medical` | 0.335 | 0.334896 | match |
| Table 2 | `T2.Mistral-Large-3.medical.sd` | 0.056 | 0.0563039 | match |
| Table 2 | `T2.Mistral-Large-3.scientific` | 0.443 | 0.443003 | match |
| Table 2 | `T2.Mistral-Large-3.scientific.sd` | 0.006 | 0.00568012 | match |
| Table 2 | `T2.gpt-5.4.legal` | 0.201 | 0.200904 | match |
| Table 2 | `T2.gpt-5.4.legal.sd` | 0.022 | 0.0220313 | match |
| Table 2 | `T2.gpt-5.4.medical` | 0.343 | 0.34314 | match |
| Table 2 | `T2.gpt-5.4.medical.sd` | 0.060 | 0.0599171 | match |
| Table 2 | `T2.gpt-5.4.scientific` | 0.422 | 0.422428 | match |
| Table 2 | `T2.gpt-5.4.scientific.sd` | 0.006 | 0.00568103 | match |
| Table A2 | `A2.legal.Phi-4|gpt-4o` | 2.5e-04 | 0.000249814 | match |
| Table A2 | `A2.legal.Phi-4|Mistral-Large-3` | 4.0e-09 | 3.97182e-09 | match |
| Table A2 | `A2.legal.Phi-4|gpt-5.4` | 2.6e-30 | 2.61357e-30 | match |
| Table A2 | `A2.legal.gpt-4o|Mistral-Large-3` | 0.014 | 0.0136966 | match |
| Table A2 | `A2.legal.gpt-4o|gpt-5.4` | 4.2e-17 | 4.2114e-17 | match |
| Table A2 | `A2.legal.Mistral-Large-3|gpt-5.4` | 5.8e-09 | 5.84048e-09 | match |
| Table A2 | `A2.medical.Phi-4|gpt-4o` | 0.146 | 0.146319 | match |
| Table A2 | `A2.medical.Phi-4|Mistral-Large-3` | 5.8e-04 | 0.000578334 | match |
| Table A2 | `A2.medical.Phi-4|gpt-5.4` | 0.003 | 0.00258097 | match |
| Table A2 | `A2.medical.gpt-4o|Mistral-Large-3` | 0.044 | 0.0439911 | match |
| Table A2 | `A2.medical.gpt-4o|gpt-5.4` | 0.112 | 0.111727 | match |
| Table A2 | `A2.medical.Mistral-Large-3|gpt-5.4` | 0.701 | 0.701153 | match |
| Table A2 | `A2.scientific.Phi-4|gpt-4o` | 0.465 | 0.465301 | match |
| Table A2 | `A2.scientific.Phi-4|Mistral-Large-3` | 9.7e-03 | 0.00973026 | match |
| Table A2 | `A2.scientific.Phi-4|gpt-5.4` | 4.8e-05 | 4.80383e-05 | match |
| Table A2 | `A2.scientific.gpt-4o|Mistral-Large-3` | 0.068 | 0.0678575 | match |
| Table A2 | `A2.scientific.gpt-4o|gpt-5.4` | 0.001 | 0.00122906 | match |
| Table A2 | `A2.scientific.Mistral-Large-3|gpt-5.4` | 0.151 | 0.151241 | match |
| App. E | `E.legal_capped.Mistral-Large-3` | 414 | 414 | match |
| App. E | `E.legal_n.Mistral-Large-3` | 592 | 592 | match |
| App. E | `E.legal_capped.gpt-5.4` | 196 | 196 | match |
| App. E | `E.legal_n.gpt-5.4` | 590 | 590 | match |
| App. E | `E.legal_capped.gpt-4o` | 73 | 73 | match |
| App. E | `E.legal_n.gpt-4o` | 588 | 588 | match |
| App. E | `E.legal_capped.Phi-4` | 0 | 0 | match |
| App. E | `E.legal_capped_efs.gpt-5.4` | 0.100 | 0.100367 | match |
| App. E | `E.legal_uncapped_efs.gpt-5.4` | 0.251 | 0.251013 | match |
| App. E | `E.legal_capped_efs.gpt-4o` | 0.267 | 0.267137 | match |
| App. E | `E.legal_uncapped_efs.gpt-4o` | 0.353 | 0.352513 | match |
| App. E | `E.mean_completion_tokens.Phi-4` | 348 | 348.236 | match |
| App. E | `E.mean_completion_tokens.Mistral-Large-3` | 456 | 455.696 | match |
| App. E | `E.mean_completion_tokens.gpt-5.4` | 399 | 398.627 | match |
| App. E | `E.band_n` | 2,937 | 2937 | match |
| App. E | `E.band_phi` | 0.487 | 0.486596 | match |
| App. E | `E.band_others` | 0.396 | 0.3959 | match |
| App. E | `E.band_U` | 979855 | 979854 | match |
| App. E | `E.band_p` | 1.6e-11 | 1.60167e-11 | match |
| App. E | `E.hosted_ols_n` | 6,971 | 6971 | match |
| App. E | `E.hosted_ols_coef` | -0.076 | -0.0758765 | match |
| App. E | `E.hosted_ols_ci_lo` | -0.103 | -0.102978 | match |
| App. E | `E.hosted_ols_ci_hi` | -0.049 | -0.0487755 | match |
| App. E | `E.hosted_ols_r2` | 0.056 | 0.0561807 | match |
| App. E | `E.hosted_loglen_r2` | 0.026 | 0.0258358 | match |
| App. C | `C.depth_over_1024` | 1,004 | 1004 | match |
| App. C | `C.depth_at_2048_cap` | 135 | 135 | match |
| App. C | `C.legal_over_1024` | 73 | 73 | match |
| App. C | `C.scientific_over_1024` | 5 | 5 | match |
| Table A4 | `A4.Q1.n` | 1,272 | 1272 | match |
| Table A4 | `A4.Q1.mean_words` | 224 | 224.327 | match |
| Table A4 | `A4.Q1.omission_per_hedge` | 0.377 | 0.377159 | match |
| Table A4 | `A4.Q1.efs` | 0.473 | 0.473498 | match |
| Table A4 | `A4.Q2.n` | 1,274 | 1274 | match |
| Table A4 | `A4.Q2.mean_words` | 339 | 338.667 | match |
| Table A4 | `A4.Q2.omission_per_hedge` | 0.320 | 0.31986 | match |
| Table A4 | `A4.Q2.efs` | 0.441 | 0.440872 | match |
| Table A4 | `A4.Q3.n` | 1,256 | 1256 | match |
| Table A4 | `A4.Q3.mean_words` | 390 | 389.729 | match |
| Table A4 | `A4.Q3.omission_per_hedge` | 0.313 | 0.31298 | match |
| Table A4 | `A4.Q3.efs` | 0.418 | 0.417945 | match |
| Table A4 | `A4.Q4.n` | 1,273 | 1273 | match |
| Table A4 | `A4.Q4.mean_words` | 441 | 441.081 | match |
| Table A4 | `A4.Q4.omission_per_hedge` | 0.292 | 0.292088 | match |
| Table A4 | `A4.Q4.efs` | 0.425 | 0.425095 | match |
| Table A4 | `A4.Q5.n` | 1,261 | 1261 | match |
| Table A4 | `A4.Q5.mean_words` | 581 | 580.757 | match |
| Table A4 | `A4.Q5.omission_per_hedge` | 0.290 | 0.28975 | match |
| Table A4 | `A4.Q5.efs` | 0.455 | 0.45549 | match |
| App. E | `E.Q5_om_min` | 0.245 | 0.245442 | match |
| App. E | `E.Q5_om_max` | 0.332 | 0.332316 | match |
| App. E | `E.len_ols_coef` | -0.060 | -0.0598604 | match |
| App. E | `E.len_ols_ci_lo` | -0.074 | -0.074426 | match |
| App. E | `E.len_ols_ci_hi` | -0.045 | -0.0452948 | match |
| App. E | `E.len_ols_p` | 9e-16 | 9.42119e-16 | match |
| App. E | `E.len_ols_r2` | 0.035 | 0.0353957 | match |
| J.5 | `J5.greedy_depth_efs` | 0.416 | 0.416034 | match |
| J.5 | `J5.greedy_omission_pct` | 43.5 | 43.5031 | match |
| J.5 | `J5.hungarian_omission_pct` | 66.3 | 66.3069 | match |
| J.5 | `J5.rank_spearman_open` | 0.936 | 0.936364 | match |
| J.5 | `J5.same_top_bottom` | 1 | 1 | match |
| J.5 | `J5.rank_spearman_hosted` | 1.000 | 1 | match |
| J.5 | `J5.identity_combinations` | 1,728 | 1728 | match |
| J.5 | `J5.identity_all_zero` | 1 | 1 | match |
| Table A11 | `A11.clinical.identity_efs` | 0.864 | 0.863531 | match |
| Table A11 | `A11.legal.identity_efs` | 0.872 | 0.872214 | match |
| Table A11 | `A11.scientific.identity_efs` | 0.788 | 0.78838 | match |
| Table A11 | `A11.clinical.12000.greedy_fpr_pct` | 66.7 | 66.6667 | match |
| Table A11 | `A11.legal.12000.greedy_fpr_pct` | 43.2 | 43.2292 | match |
| Table A11 | `A11.scientific.12000.greedy_fpr_pct` | 69.8 | 69.7917 | match |
| Table A11 | `A11.clinical.observed_greedy` | 0.414 | 0.414264 | match |
| Table A11 | `A11.legal.observed_greedy` | 0.361 | 0.361453 | match |
| Table A11 | `A11.scientific.observed_greedy` | 0.395 | 0.394872 | match |
| J.5 | `A11.clinical.3000.greedy_fpr_pct` | 18.8 | 18.75 | match |
| J.5 | `A11.clinical.6000.greedy_fpr_pct` | 50.5 | 50.5208 | match |
| J.5 | `A11.clinical.12000.greedy_spurious` | 452 | 452 | match |
| J.6 | `J6.floor_pct.omission` | 0.0 | 0 | match |
| J.6 | `J6.floor_pct.fabrication` | 0.0 | 0 | match |
| J.6 | `J6.floor_pct.attenuation` | 6.7 | 6.68187 | match |
| J.6 | `J6.floor_pct.scope_shift` | 12.9 | 12.9249 | match |
| J.6 | `J6.floor_pct.severity_warp` | 31.6 | 31.5945 | match |
| J.6 | `J6.floor_pct.escalation` | 51.6 | 51.5844 | match |
| J.6 | `J6.omission_plus_fabrication_pct` | 74.7 | 74.7317 | match |
| App. D / J.4 | `J4.markers_per_doc.seed123` | 15.8 | 15.7865 | match |
| J.4 | `J4.markers_per_doc.seed0` | 23.5 | 23.4511 | match |
| J.4 | `J4.markers_per_doc.seed1` | 22.4 | 22.3913 | match |
| J.4 | `J4.markers_per_doc.3seeds` | 20.5 | 20.475 | match |
| App. D | `D.contextual_per_doc` | 6.0 | 6 | match |
| App. D | `D.contextual_reduction_pct` | 62 | 61.9927 | match |
| J.1 / J.4 | `J4.severity_temporal_pct` | 74.7 | 74.6618 | match |
| J.1 | `J1.strength_zero_pct` | 84.3 | 84.3286 | match |
| J.4 | `J4.scientific_quantitative_pct` | 44 | 43.8466 | match |
| 6.1 / J.5 | `J5.sweep.efs_min` | 0.441 | 0.441161 | match |
| 6.1 / J.5 | `J5.sweep.efs_max` | 0.449 | 0.448737 | match |
| J.5 | `J5.sweep.0.30.efs` | 0.442 | 0.442095 | match |
| J.5 | `J5.sweep.0.20.omission_pct` | 56.3 | 56.2507 | match |
| J.5 | `J5.sweep.0.30.omission_pct` | 61.2 | 61.2061 | match |
| J.5 | `J5.sweep.0.50.omission_pct` | 74.3 | 74.2636 | match |
| J.7 / Table A12 | `J7.rho_mean` | -0.891 | -0.891108 | match |
| J.5 | `J7.rho_std` | 0.006 | 0.00560001 | match |
| Table A12 | `J7.mean_efs.0.00` | 1.000 | 1 | match |
| Table A12 | `J7.mean_efs.0.25` | 0.830 | 0.830183 | match |
| Table A12 | `J7.mean_efs.0.50` | 0.644 | 0.644491 | match |
| Table A12 | `J7.mean_efs.0.75` | 0.443 | 0.442942 | match |
| Table A12 | `J7.mean_efs.1.00` | 0.249 | 0.248828 | match |
| J.5 | `J5.gt_greedy.rho_mean` | -0.721 | -0.720956 | match |
| J.5 | `J5.gt_greedy.rho_std` | 0.009 | 0.00895305 | match |
| J.7 | `J7.claude.severity_vs_fraction_rho` | 0.650 | 0.649986 | match |
| 6.1 / J.4 | `J4.clinical_3seed_efs` | 0.366 | 0.366429 | match |
| J.4 | `J4.markers_per_doc` | 6.2 | 6.1875 | match |
| 6.1 / J.4 | `J4.rank_spearman_vs_unrestricted` | 0.873 | 0.872727 | match |
| J.4 | `J4.rank_spearman_p` | < 0.001 | 0.000454615 | match |
| J.4 | `J4.rank.biomistral-7b` | 1 | 1 | match |
| J.4 | `J4.rank.medgemma-27b` | 11 | 11 | match |
| J.4 | `J4.omission_pct` | 60.3 | 60.2666 | match |
| 6.2 / J.4 | `J4.escalation_pct` | 21.3 | 21.2545 | match |
| J.4 | `J4.unrestricted.crossseed_std_max` | 0.020 | 0.0204843 | match |
| J.4 | `J4.crossseed_std_max` | 0.067 | 0.0673968 | match |
| J.4 | `J4.matched.clinical_efs` | 0.456 | 0.455507 | match |
| J.4 | `J4.matched.scientific_efs` | 0.415 | 0.415028 | match |
| J.4 | `J4.matched.legal_efs` | 0.359 | 0.359091 | match |
| J.4 | `J4.matched.kruskal_H` | 44.8 | 44.7807 | match |
| J.4 | `J4.matched.kruskal_p` | 1.9e-10 | 1.888e-10 | match |
| J.3 | `J3.n_notes` | 60 | 60 | match |
| J.3 | `J3.markers_per_note` | 16.9 | 16.9167 | match |
| J.3 | `J3.identity_notes_with_distortion` | 0 | 0 | match |
| J.3 | `J3.greedy_identity_mean_efs` | 0.823 | 0.823317 | match |
| J.3 | `J3.greedy_identity_fpr_pct` | 80.0 | 80 | match |
| J.3 | `J3.deletion_steps` | 919 | 919 | match |
| J.3 | `J3.deletion_violations` | 0 | 0 | match |
| J.3 | `J3.efs_after_all_deletions` | 0.340 | 0.33967 | match |
| J.3 | `J3.scale_max_abs_delta` | 0.000 | 0 | match |
| J.3 | `J3.graft_notes_decreased` | 59 | 59 | match |
| J.3 | `J3.graft_efs_after` | 0.907 | 0.907337 | match |
| J.3 | `J3.certain_notes_decreased` | 1 | 1 | match |
| J.3 | `J3.markers_never_in_claim_pct` | 39 | 39.327 | match |
| J.3 | `J3.efs0.implicated_pct_pooled` | 68 | 68.3428 | match |
| J.3 | `A10.markers_per_note` | 19.1 | 19.1 | match |
| J.3 | `A10.notes_shared_with_J3` | 53 | 53 | match |
| Table A10 | `A10.identity.efs` | 1.000 | 1 | match |
| Table A10 | `A10.identity.hedge_count_ratio` | 1.000 | 1 | match |
| Table A10 | `A10.identity.rougeL` | 1.000 | 1 | match |
| Table A10 | `A10.reorder.efs` | 0.998 | 0.997553 | match |
| Table A10 | `A10.reorder.hedge_count_ratio` | 1.000 | 1 | match |
| Table A10 | `A10.reorder.rougeL` | 0.334 | 0.333503 | match |
| Table A10 | `A10.delete_0.50.efs` | 0.670 | 0.669798 | match |
| Table A10 | `A10.delete_0.50.hedge_count_ratio` | 0.498 | 0.498111 | match |
| Table A10 | `A10.delete_0.50.rougeL` | 0.996 | 0.99636 | match |
| Table A10 | `A10.relocate.efs` | 0.806 | 0.806457 | match |
| Table A10 | `A10.relocate.hedge_count_ratio` | 1.000 | 1 | match |
| Table A10 | `A10.relocate.rougeL` | 0.997 | 0.996726 | match |
| Table A10 | `A10.add_hedge.efs` | 0.935 | 0.934739 | match |
| Table A10 | `A10.add_hedge.hedge_count_ratio` | 1.218 | 1.21801 | match |
| Table A10 | `A10.add_hedge.rougeL` | 0.998 | 0.998395 | match |
| Table A10 | `A10.fabricate.efs` | 0.931 | 0.93106 | match |
| Table A10 | `A10.fabricate.hedge_count_ratio` | 1.070 | 1.06954 | match |
| Table A10 | `A10.fabricate.rougeL` | 0.993 | 0.993462 | match |
| J.3 | `A10.deletion_spearman.efs` | -0.897 | -0.897013 | match |
| J.3 | `A10.deletion_spearman.hedge_count_ratio` | -0.988 | -0.988213 | match |
| J.3 | `A10.deletion_spearman.rougeL` | -0.864 | -0.863945 | match |
| F.2 | `F2.identity.efs` | 1.000 | 1 | match |
| 6.3 / F.2 | `F2.dehedge.efs` | 0.268 | 0.267696 | match |
| 6.3 / F.2 | `F2.dehedge.rouge_l` | 0.994 | 0.993843 | match |
| F.2 | `F2.content_cut.efs` | 0.757 | 0.756613 | match |
| F.2 | `F2.content_cut.rouge_l` | 0.676 | 0.675875 | match |
| J.5 | `F2.dehedge.greedy_efs` | 0.270 | 0.270316 | match |
| 6.3 | `S6.3.n` | 6,336 | 6336 | match |
| 6.3 | `S6.3.rouge_r` | 0.107 | 0.107152 | match |
| 6.3 | `S6.3.rouge_p` | 1.2e-17 | 1.20837e-17 | match |
| 6.3 | `S6.3.rouge_r2` | 0.011 | 0.0114815 | match |
| 6.3 | `S6.3.bertscore_r` | -0.101 | -0.101262 | match |
| 6.3 | `S6.3.bertscore_p` | 6.5e-16 | 6.49969e-16 | match |
| F.3 | `F3.n` | 200 | 200 | match |
| F.3 | `F3.efs_before` | 0.493 | 0.493459 | match |
| F.3 | `F3.efs_after` | 0.832 | 0.831736 | match |
| F.3 | `F3.dist_reduction_pct` | 66 | 66.2393 | match |
| F.3 | `F3.greedy_efs_before` | 0.458 | 0.45838 | match |
| F.3 | `F3.greedy_efs_after` | 0.732 | 0.73195 | match |
| F.3 | `F3.greedy_dist_reduction_pct` | 50 | 50.2541 | match |
| F.4 | `F4.legal.coef_is_model` | 0.025 | 0.0245095 | match |
| F.4 | `F4.legal.coef_p` | 0.057 | 0.0570175 | match |
| F.4 | `F4.legal.coef_ci_lo` | -0.001 | -0.000732348 | match |
| F.4 | `F4.legal.coef_ci_hi` | 0.050 | 0.0497513 | match |
| F.4 | `F4.legal.band_n_human` | 180 | 180 | match |
| F.4 | `F4.legal.band_n_model` | 1,152 | 1152 | match |
| F.4 | `F4.legal.band_human_harden` | 0.082 | 0.0820417 | match |
| F.4 | `F4.legal.band_model_harden` | 0.107 | 0.106717 | match |
| F.4 | `F4.legal.band_mannwhitney_p` | 0.024 | 0.0244313 | match |
| F.4 | `F4.scientific.coef_is_model` | 0.117 | 0.116988 | match |
| F.4 | `F4.scientific.coef_p` | 1.6e-11 | 1.5549e-11 | match |
| F.4 | `F4.scientific.human_compression` | 0.021 | 0.0208901 | match |
| F.4 | `F4.scientific.model_compression` | 0.317 | 0.316516 | match |
| F.4 | `F4.scientific.band_n_human` | 4 | 4 | match |
| F.5 | `F5.proxy.n` | 150 | 150 | match |
| F.5 | `F5.proxy.any_change_pct` | 82.7 | 82.6667 | match |
| F.5 | `F5.proxy.urgency_changed_pct` | 65 | 65.3333 | match |
| F.5 | `F5.proxy.action_changed_pct` | 46 | 46 | match |
| F.5 / J.8 | `F5.identity_floor.any_change_pct` | 27.3 | 27.3333 | match |
| F.5 | `F5.paired.any.mcnemar_exact_p` | 1.1e-20 | 1.11122e-20 | match |
| F.5 | `F5.harm.n` | 150 | 150 | match |
| F.5 | `F5.harm.pointbiserial_r` | 0.010 | 0.00954471 | match |
| F.5 | `F5.harm.pointbiserial_p` | 0.908 | 0.907713 | match |
| F.5 | `F5.harm.bin1.change_pct` | 70.8 | 70.8333 | match |
| F.5 | `F5.harm.bin2.change_pct` | 75.0 | 75 | match |
| F.5 | `F5.harm.bin3.change_pct` | 82.1 | 82.1429 | match |
| F.5 | `F5.harm.bin4.change_pct` | 73.1 | 73.0769 | match |
| F.5 | `F5.harm.bin5.change_pct` | 75.0 | 75 | match |
| F.6 | `F6.n` | 185 | 185 | match |
| F.6 | `F6.dehedge_change_pct` | 15.7 | 15.6757 | match |
| F.6 | `F6.identity_floor_pct` | 1.1 | 1.08108 | match |
| F.6 | `F6.excess_pts` | 14.6 | 14.5946 | match |
| F.6 | `F6.bootstrap_ci_lo` | 9.2 | 9.18919 | match |
| F.6 | `F6.bootstrap_ci_hi` | 20.0 | 20 | match |
| F.6 | `F6.mcnemar_b_dehedge_only` | 28 | 28 | match |
| F.6 | `F6.mcnemar_c_identity_only` | 1 | 1 | match |
| F.6 | `F6.mcnemar_exact_p` | 1.1e-07 | 1.11759e-07 | match |
| F.6 | `F6.left_padded_identity_floor_pct` | 12.5 | 12.5 | match |
| App. I | `I.positives.n` | 242 | 242 | match |
| App. I | `I.positives.per_model_min` | 22 | 22 | match |
| App. I | `I.positives.per_model_max` | 22 | 22 | match |
| App. I | `I.positives.judge70b_flagged` | 242 | 242 | match |
| App. I / Table A8 | `I.positives.pearson_r` | 0.186 | 0.185561 | match |
| App. I / Table A8 | `I.positives.pearson_p` | 0.004 | 0.00376858 | match |
| Table A8 | `A8.truncation.judge70b_flagged` | 139 | 139 | match |
| App. I | `I.truncation.judge70b_flag_pct` | 57.4 | 57.438 | match |
| 6.6 / App. I | `I.identity.judge70b.flagged` | 1 | 1 | match |
| App. I | `I.identity.judge70b.flag_pct` | 0.4 | 0.413223 | match |
| App. I | `I.identity.judge70b.wilson_lo_pct` | 0.1 | 0.0729813 | match |
| App. I | `I.identity.judge70b.wilson_hi_pct` | 2.3 | 2.30313 | match |
| App. I | `I.identity.judge70b.parse_fail` | 0 | 0 | match |
| App. I | `I.identity.judge70b.rated_same` | 242 | 242 | match |
| App. I | `I.identity.claude.flag_pct` | 0.0 | 0 | match |
| App. I | `I.identity.claude.wilson_hi_pct` | 1.6 | 1.56258 | match |
| App. I | `I.truncation.mean_source_chars` | 10210 | 10209.7 | match |
| App. I | `I.truncation.hedges_retained_pct` | 24.7 | 24.6684 | match |
| App. I | `I.truncation.pairs_losing_hedge` | 236 | 236 | match |
| Table A8 | `A8.truncation.pairs_losing_hedge_pct` | 97.5 | 97.5207 | match |
| App. I / Table A8 | `A8.truncation.pipeline_greedy_flag_pct` | 89.7 | 89.6694 | match |
| App. I / Table A8 | `A8.identity.pipeline_greedy_fpr_pct` | 27.7 | 27.686 | match |
| App. I | `I.positive_control.P.flag_pct` | 98.3 | 98.3333 | match |
| App. I | `I.positive_control.P.mean_severity` | 2.833 | 2.83333 | match |
| App. I | `I.positive_control.P.distinct_explanations` | 60 | 60 | match |
| App. I | `I.positive_control.D.flag_pct` | 60.0 | 60 | match |
| App. I | `I.positive_control.T.flag_pct` | 1.7 | 1.66667 | match |
| App. I | `I.positive_control.I.flag_pct` | 0.0 | 0 | match |
| 6.6 | `S6.6.n_submitted` | 750 | 750 | match |
| 6.6 | `S6.6.strict.n` | 541 | 541 | match |
| 6.6 | `S6.6.strict.more_certain_pct` | 96.3 | 96.3031 | match |
| 6.6 | `S6.6.strict.clinical.more_certain_pct` | 98.3 | 98.3471 | match |
| 6.6 | `S6.6.strict.scientific.more_certain_pct` | 94.6 | 94.6488 | match |
| 6.6 | `S6.6.strict.less_certain_pct` | 2.8 | 2.77264 | match |
| 6.6 / J.8 | `J8.recovered.more_certain_pct` | 97.1 | 97.0667 | match |
| J.8 | `J8.recovered.clinical.more_certain_pct` | 98.7 | 98.6667 | match |
| J.8 | `J8.recovered.scientific.more_certain_pct` | 94.7 | 94.6667 | match |
| J.8 | `J8.recovered.less_certain_pct` | 2.0 | 2 | match |
| J.8 | `J8.code_fenced_recovered` | 209 | 209 | match |
| 6.7 | `J2.control.n` | 389 | 389 | match |
| J.2 | `J2.control.n_reports` | 161 | 161 | match |
| 6.7 | `J2.control.source_preserved_pct` | 83.8 | 83.8046 | match |
| 6.7 | `J2.control.model_preserved_pct` | 54.5 | 54.4987 | match |
| 6.7 | `J2.control.drop_pts` | 29.3 | 29.3059 | match |
| 6.7 | `J2.control.drop_p` | 1.2e-25 | 1.22371e-25 | match |
| 6.7 | `J2.control.drop_ci_lo` | 23.3 | 23.2613 | match |
| 6.7 | `J2.control.drop_ci_hi` | 35.3 | 35.3427 | match |
| J.2 | `J2.control.omitted_pct` | 3.9 | 3.85604 | match |
| 6.7 | `J2.both_mentioned.n` | 329 | 329 | match |
| 6.7 | `J2.both_mentioned.source_pct` | 14.3 | 14.2857 | match |
| 6.7 | `J2.both_mentioned.model_pct` | 36.5 | 36.4742 | match |
| J.2 | `J2.both_mentioned.p` | 2.5e-16 | 2.52486e-16 | match |
| 6.7 / J.2 | `J2.survival` | 0.510 | 0.510294 | match |
| 6.7 | `J2.survival_ci_lo` | 0.465 | 0.464743 | match |
| 6.7 | `J2.survival_ci_hi` | 0.555 | 0.555346 | match |
| J.2 | `J2.survival.biomistral-7b` | 0.512 | 0.511765 | match |
| J.2 | `J2.survival.llama-3.1-8b` | 0.540 | 0.539706 | match |
| J.2 | `J2.survival.med42-8b` | 0.513 | 0.513235 | match |
| J.2 | `J2.survival.mistral-7b` | 0.476 | 0.476471 | match |
| J.2 | `J2.n_judgements` | 2,720 | 2720 | match |
| J.2 | `J2.paired_drop_p_max` | < 0.000200001 | 0.000195126 | match |
| J.2 | `J2.mean_chars.source` | 795 | 794.666 | match |
| J.2 | `J2.mean_chars.model` | 367 | 366.936 | match |
| Table A9 | `A9.preserved.primary.n` | 389 | 389 | match |
| Table A9 | `A9.preserved.primary.source_pct` | 83.8 | 83.8046 | match |
| Table A9 | `A9.preserved.primary.model_pct` | 54.5 | 54.4987 | match |
| Table A9 | `A9.preserved.primary.delta_pts` | -29.3 | -29.3059 | match |
| Table A9 | `A9.preserved.primary.p` | 1.2e-25 | 1.22371e-25 | match |
| Table A9 | `A9.preserved.primary.ci_lo` | -35.3 | -35.3427 | match |
| Table A9 | `A9.preserved.primary.ci_hi` | -23.3 | -23.2613 | match |
| Table A9 | `A9.preserved.secondary.n` | 327 | 327 | match |
| Table A9 | `A9.preserved.secondary.source_pct` | 89.0 | 88.9908 | match |
| Table A9 | `A9.preserved.secondary.model_pct` | 57.5 | 57.4924 | match |
| Table A9 | `A9.preserved.secondary.delta_pts` | -31.5 | -31.4985 | match |
| Table A9 | `A9.preserved.secondary.p` | 6.4e-25 | 6.36731e-25 | match |
| Table A9 | `A9.preserved.secondary.ci_lo` | -38.0 | -37.9825 | match |
| Table A9 | `A9.preserved.secondary.ci_hi` | -25.0 | -25 | match |
| Table A9 | `A9.escalated.primary.n` | 389 | 389 | match |
| Table A9 | `A9.escalated.primary.source_pct` | 12.3 | 12.3393 | match |
| Table A9 | `A9.escalated.primary.model_pct` | 30.8 | 30.8483 | match |
| Table A9 | `A9.escalated.primary.delta_pts` | 18.5 | 18.509 | match |
| Table A9 | `A9.escalated.primary.p` | 1.3e-15 | 1.2794e-15 | match |
| Table A9 | `A9.escalated.primary.ci_lo` | 13.4 | 13.3879 | match |
| Table A9 | `A9.escalated.primary.ci_hi` | 23.5 | 23.5453 | match |
| Table A9 | `A9.escalated.secondary.n` | 327 | 327 | match |
| Table A9 | `A9.escalated.secondary.source_pct` | 6.4 | 6.42202 | match |
| Table A9 | `A9.escalated.secondary.model_pct` | 25.4 | 25.3823 | match |
| Table A9 | `A9.escalated.secondary.delta_pts` | 19.0 | 18.9602 | match |
| Table A9 | `A9.escalated.secondary.p` | 6.4e-15 | 6.38773e-15 | match |
| Table A9 | `A9.escalated.secondary.ci_lo` | 13.8 | 13.7703 | match |
| Table A9 | `A9.escalated.secondary.ci_hi` | 24.2 | 24.2038 | match |
| Table A9 | `A9.hardened_mentioned.primary.n_source` | 374 | 374 | match |
| Table A9 | `A9.hardened_mentioned.primary.n_model` | 332 | 332 | match |
| Table A9 | `A9.hardened_mentioned.primary.source_pct` | 12.8 | 12.8342 | match |
| Table A9 | `A9.hardened_mentioned.primary.model_pct` | 36.1 | 36.1446 | match |
| Table A9 | `A9.hardened_mentioned.primary.delta_pts` | 23.3 | 23.3104 | match |
| 6.7 / Table A9 | `A9.hardened_mentioned.primary.ratio` | 2.816 | 2.81627 | match |
| Table A9 | `A9.hardened_mentioned.primary.ratio_ci_lo` | 2.130 | 2.12961 | match |
| Table A9 | `A9.hardened_mentioned.primary.ratio_ci_hi` | 3.954 | 3.95428 | match |
| Table A9 | `A9.hardened_mentioned.primary.delta_ci_lo` | 17.5 | 17.5169 | match |
| Table A9 | `A9.hardened_mentioned.primary.delta_ci_hi` | 29.2 | 29.197 | match |
| 6.7 / J.1 | `J1.radgraph.pooled.epistemic.n` | 3,622 | 3622 | match |
| 6.7 / J.1 | `J1.radgraph.pooled.epistemic.precision` | 0.570 | 0.569519 | match |
| 6.7 / J.1 | `J1.radgraph.pooled.epistemic.recall` | 0.609 | 0.608571 | match |
| 6.7 / J.1 | `J1.radgraph.pooled.epistemic.f1` | 0.588 | 0.588398 | match |
| J.1 | `J1.radgraph.test_r1.epistemic.f1` | 0.606 | 0.606061 | match |
| J.1 | `J1.radgraph.test_r2.epistemic.f1` | 0.691 | 0.690909 | match |
| J.1 | `J1.radgraph.pooled.all_spans.precision` | 0.170 | 0.169834 | match |
| J.1 | `J1.radgraph.interrater.n_sentences` | 1,011 | 1011 | match |
| J.1 | `J1.radgraph.interrater.agreement` | 0.989 | 0.98912 | match |
| J.1 | `J1.radgraph.interrater.kappa` | 0.879 | 0.878582 | match |
| 6.7 / J.1 | `J1.radgraph.interrater.f1` | 0.884 | 0.884211 | match |
| J.1 | `J1.nubes.n_files` | 615 | 615 | match |
| J.1 | `J1.nubes.n_cues` | 2,795 | 2795 | match |
| J.1 | `J1.nubes.n_forms` | 321 | 321 | match |
| J.1 | `J1.nubes.handbuilt.cue_recall_mean` | 0.448 | 0.447745 | match |
| J.1 | `J1.nubes.handbuilt.cue_recall_std` | 0.019 | 0.0186646 | match |
| J.1 | `J1.nubes.handbuilt.cue_recall_min` | 0.420 | 0.41979 | match |
| J.1 | `J1.nubes.handbuilt.precision_mean` | 0.824 | 0.824002 | match |
| J.1 | `J1.nubes.handbuilt.recall_mean` | 0.487 | 0.487436 | match |
| 6.7 / J.1 | `J1.nubes.handbuilt.f1_mean` | 0.612 | 0.612439 | match |
| J.1 | `J1.nubes.derived100.cue_recall_mean` | 0.909 | 0.909351 | match |
| J.1 | `J1.nubes.derived100.cue_recall_std` | 0.008 | 0.00827015 | match |
| J.1 | `J1.nubes.derived100.precision_mean` | 0.250 | 0.250468 | match |
| J.1 | `J1.nubes.derived100.f1_mean` | 0.394 | 0.393806 | match |
| Table A13 | `A13.vs_human.claude-sonnet-4-6.ac1` | 0.947 | 0.947438 | match |
| Table A13 | `A13.vs_human.llama-3.3-70b.ac1` | 0.942 | 0.941945 | match |
| Table A13 | `A13.vs_human.gpt-oss-20b.ac1` | 0.934 | 0.934426 | match |
| Table A13 | `A13.vs_human.gemma-4-31b.ac1` | 0.559 | 0.558824 | match |
| Table A13 | `A13.vs_human.gpt-oss-20b.n` | 18 | 18 | match |
| Table A13 | `A13.vs_pipeline.claude-sonnet-4-6.ac1` | 0.907 | 0.907037 | match |
| Table A13 | `A13.vs_pipeline.llama-3.3-70b.ac1` | 0.921 | 0.92128 | match |
| Table A13 | `A13.vs_pipeline.gpt-oss-20b.ac1` | 0.792 | 0.792085 | match |
| Table A13 | `A13.vs_pipeline.gemma-4-31b.ac1` | 0.607 | 0.606905 | match |
| Table A13 | `A13.vs_pipeline.gpt-oss-20b.n` | 139 | 139 | match |
| Table A13 | `A13.vs_pipeline.claude-sonnet-4-6.n` | 164 | 164 | match |
| J.9 | `A13.ai_vs_ai.claude-sonnet-4-6|llama-3.3-70b.ac1` | 0.948 | 0.948155 | match |
| App. K | `K.severity_qwk.mean` | 0.430 | 0.429934 | match |
| App. K | `K.severity_qwk.R1-R2` | 0.554 | 0.553571 | match |
| App. K | `K.severity_qwk.R1-R3` | 0.543 | 0.542683 | match |
| App. K | `K.severity_qwk.R2-R3` | 0.194 | 0.193548 | match |
| App. K | `K.severity_within1_pct` | 70 | 70 | match |
| App. K | `K.fleiss.fabrication` | 0.603 | 0.603175 | match |
| App. K | `K.positives.fabrication.R1` | 13 | 13 | match |
| App. K | `K.positives.fabrication.R2` | 14 | 14 | match |
| App. K | `K.positives.fabrication.R3` | 15 | 15 | match |
| App. K | `K.positives.omission.R1` | 19 | 19 | match |
| App. K | `K.positives.omission.R2` | 6 | 6 | match |
| App. K | `K.any_distortion.raw_pairwise` | 0.93 | 0.933333 | match |
| App. K | `K.any_distortion.ac1_mean_pairwise` | 0.926 | 0.926335 | match |
| App. K | `K.pearson_r` | -0.034 | -0.0343498 | match |
| App. K | `K.pearson_boot_lo` | -0.523 | -0.522784 | match |
| App. K | `K.pearson_boot_hi` | 0.457 | 0.456724 | match |
| App. K | `K.spearman_rho` | -0.070 | -0.0704998 | match |
| App. K | `K.spearman_boot_lo` | -0.554 | -0.554492 | match |
| App. K | `K.spearman_boot_hi` | 0.440 | 0.439726 | match |
| Table A15 | `M.llama-3.1-8b.baseline_ratio` | 24.818 | 24.808 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.llama-3.1-8b.hedge_ratio` | 3.449 | 3.45282 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.llama-3.1-8b.reduction_pct` | 89.7 | 89.6975 | match |
| Table A15 | `M.mistral-7b.baseline_ratio` | 83.080 | 83.0112 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.mistral-7b.hedge_ratio` | 3.574 | 3.57506 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.mistral-7b.reduction_pct` | 96.9 | 96.8601 | match |
| Table A15 | `M.med42-8b.baseline_ratio` | 28.696 | 28.718 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.med42-8b.hedge_ratio` | 3.673 | 3.67983 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.med42-8b.reduction_pct` | 90.3 | 90.3318 | match |
| Table A15 | `M.gemma-3-4b.baseline_ratio` | 82.334 | 81.4786 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.gemma-3-4b.hedge_ratio` | 7.605 | 7.60049 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| Table A15 | `M.gemma-3-4b.reduction_pct` | 91.9 | 91.7984 | approximate (float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions) |
| 6.8 / M | `M.llama-3.1-8b.hedge_frac_assertive_pct` | 90.0 | 90 | match |
| M | `M.llama-3.1-8b.baseline_frac_assertive_pct` | 99.8 | 99.8 | match |
| M | `M.llama-3.1-8b.n_assertive_ids` | 20 | 20 | match |
| M | `M.llama-3.1-8b.n_hedge_ids` | 22 | 22 | match |
| M | `M.length_match_p` | 0.501 | 0.500635 | match |
| Table A14 | `A14.PubMedBERT.legal.acc` | 49.2 | 49.1701 | match |
| Table A14 | `A14.PubMedBERT.scientific.acc` | 58.8 | 58.7814 | match |
| Table A14 | `A14.PubMedBERT.clinical_val.acc` | 87.6 | 87.65 | match |
| Table A14 | `A14.PubMedBERT.legal.binary_acc` | 55.2 | 55.1867 | match |
| Table A14 | `A14.PubMedBERT.scientific.binary_acc` | 74.6 | 74.552 | match |
| Table A14 | `A14.PubMedBERT.clinical_val.binary_acc` | 96.6 | 96.6 | match |
| Table A14 | `A14.RoBERTa-base.legal.acc` | 64.5 | 64.5228 | match |
| Table A14 | `A14.RoBERTa-base.scientific.acc` | 71.0 | 70.9677 | match |
| Table A14 | `A14.RoBERTa-base.clinical_val.acc` | 87.9 | 87.95 | match |
| Table A14 | `A14.RoBERTa-base.legal.binary_acc` | 74.9 | 74.8963 | match |
| Table A14 | `A14.RoBERTa-base.scientific.binary_acc` | 87.5 | 87.4552 | match |
| Table A14 | `A14.RoBERTa-base.clinical_val.binary_acc` | 97.0 | 97 | match |
| Table A14 | `A14.RoBERTa-large.legal.acc` | 69.9 | 69.917 | match |
| Table A14 | `A14.RoBERTa-large.scientific.acc` | 73.5 | 73.4767 | match |
| Table A14 | `A14.RoBERTa-large.clinical_val.acc` | 87.8 | 87.8 | match |
| Table A14 | `A14.RoBERTa-large.legal.binary_acc` | 84.0 | 84.0249 | match |
| Table A14 | `A14.RoBERTa-large.scientific.binary_acc` | 93.2 | 93.19 | match |
| Table A14 | `A14.RoBERTa-large.clinical_val.binary_acc` | 97.0 | 97 | match |
| Table A14 | `A14.SciBERT.legal.acc` | 63.1 | 63.0705 | match |
| Table A14 | `A14.SciBERT.scientific.acc` | 64.5 | 64.5161 | match |
| Table A14 | `A14.SciBERT.clinical_val.acc` | 88.0 | 88 | match |
| Table A14 | `A14.SciBERT.legal.binary_acc` | 72.8 | 72.8216 | match |
| Table A14 | `A14.SciBERT.scientific.binary_acc` | 80.6 | 80.6452 | match |
| Table A14 | `A14.SciBERT.clinical_val.binary_acc` | 96.9 | 96.85 | match |
| Table A14 | `A14.RoBERTa-base.legal.n` | 482 | 482 | match |
| Table A14 | `A14.RoBERTa-base.scientific.n` | 279 | 279 | match |
| Table A16 | `A16.RoBERTa-base.acc_mean` | 88.7 | 88.7242 | match |
| Table A16 | `A16.RoBERTa-base.acc_std` | 0.2 | 0.171316 | match |
| Table A16 | `A16.RoBERTa-base.macro_f1_mean` | 0.836 | 0.835681 | match |
| Table A16 | `A16.RoBERTa-base.f1_escalation` | 0.814 | 0.814134 | match |
| Table A16 | `A16.RoBERTa-base.f1_attenuation` | 0.853 | 0.853002 | match |
| Table A16 | `A16.RoBERTa-base.f1_scope_shift` | 0.736 | 0.735959 | match |
| Table A16 | `A16.RoBERTa-base.f1_severity_warp` | 0.620 | 0.619545 | match |
| Table A16 | `A16.PubMedBERT.acc_mean` | 88.8 | 88.8173 | match |
| Table A16 | `A16.PubMedBERT.acc_std` | 0.1 | 0.13965 | match |
| Table A16 | `A16.PubMedBERT.macro_f1_mean` | 0.835 | 0.835481 | match |
| Table A16 | `A16.PubMedBERT.f1_escalation` | 0.817 | 0.816759 | match |
| Table A16 | `A16.PubMedBERT.f1_attenuation` | 0.863 | 0.863418 | match |
| Table A16 | `A16.PubMedBERT.f1_scope_shift` | 0.735 | 0.735271 | match |
| Table A16 | `A16.PubMedBERT.f1_severity_warp` | 0.612 | 0.612477 | match |
| Table A16 | `A16.SciBERT.acc_mean` | 88.8 | 88.8352 | match |
| Table A16 | `A16.SciBERT.acc_std` | 0.1 | 0.0529905 | match |
| Table A16 | `A16.SciBERT.macro_f1_mean` | 0.838 | 0.837577 | match |
| Table A16 | `A16.SciBERT.f1_escalation` | 0.817 | 0.816731 | match |
| Table A16 | `A16.SciBERT.f1_attenuation` | 0.864 | 0.863999 | match |
| Table A16 | `A16.SciBERT.f1_scope_shift` | 0.738 | 0.737722 | match |
| Table A16 | `A16.SciBERT.f1_severity_warp` | 0.615 | 0.615021 | match |
| Table A16 | `A16.RoBERTa-large.acc_mean` | 88.5 | 88.4592 | match |
| Table A16 | `A16.RoBERTa-large.acc_std` | 0.1 | 0.0715257 | match |
| Table A16 | `A16.RoBERTa-large.macro_f1_mean` | 0.833 | 0.832505 | match |
| Table A16 | `A16.RoBERTa-large.f1_escalation` | 0.812 | 0.811823 | match |
| Table A16 | `A16.RoBERTa-large.f1_attenuation` | 0.848 | 0.847805 | match |
| Table A16 | `A16.RoBERTa-large.f1_scope_shift` | 0.727 | 0.727386 | match |
| Table A16 | `A16.RoBERTa-large.f1_severity_warp` | 0.621 | 0.621059 | match |
| J.1 | `J1.bioscope.clinical.n_sentences` | 11,871 | 11871 | match |
| 6.7 / J.1 | `J1.bioscope.scientific.f1` | 0.559 | 0.559 | match |
| J.1 | `J1.bioscope.scientific.precision` | 0.628 | 0.628 | match |
| 6.7 / J.1 | `J1.bioscope.clinical.f1` | 0.395 | 0.395 | match |
| J.1 | `J1.bioscope.clinical.precision` | 0.533 | 0.533 | match |
| J.1 | `J1.bioscope.clinical.recall` | 0.313 | 0.313 | match |
| J.1 | `J1.bioscope.contextual.f1` | 0.082 | 0.082 | match |
| J.1 | `J1.bioscope.contextual.precision` | 0.405 | 0.405 | match |
| J.1 | `J1.bioscope.contextual.recall` | 0.046 | 0.046 | match |
| F.7 | `F7.lexicon_patterns.en` | 17 | 17 | match |
| F.7 | `F7.lexicon_patterns.es` | 19 | 19 | match |
| F.7 | `F7.lexicon_patterns.fr` | 16 | 16 | match |
| F.7 | `F7.gemma-4-4b.en.retention` | 0.955 | 0.954911 | match |
| F.7 | `F7.gemma-4-4b.es.retention` | 0.531 | 0.530682 | match |
| F.7 | `F7.gemma-4-4b.fr.retention` | 0.411 | 0.410638 | match |
| F.7 | `F7.llama-3.1-8b.en.retention` | 1.541 | 1.54117 | match |
| F.7 | `F7.llama-3.1-8b.es.retention` | 0.562 | 0.56246 | match |
| F.7 | `F7.llama-3.1-8b.fr.retention` | 0.434 | 0.434192 | match |
