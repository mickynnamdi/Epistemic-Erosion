"""Values printed in the paper, used by scripts/make_tables.py.

Each entry: check_id -> (kind, expected, where)
  ("dp", x, k)   value rounded to k decimals must equal x
  ("sci", x, k)  scientific notation: value must equal x to k significant digits
  ("lt", x)      value < x          ("ge", x)  value >= x
  ("int", n)     integer equality
`where` is the paper location. Check ids match the keys produced in scripts/analysis/*.
"""

T1 = {  # model: efs, std, dist/gen, esc%, att%, omi%, fab%, scope%, sevw%
    "biomistral-7b": (.493, .002, 10.423, 7.6, 0.3, 82.9, 3.2, 3.3, 2.7),
    "llama-3.1-8b":  (.481, .005, 10.611, 10.6, 1.5, 72.3, 4.9, 6.4, 4.2),
    "openbio-70b":   (.478, .001, 10.660, 10.4, 1.2, 75.3, 4.1, 5.0, 3.9),
    "mistral-7b":    (.478, .002, 10.619, 11.0, 1.3, 73.4, 4.5, 5.6, 4.2),
    "med42-8b":      (.477, .006, 10.521, 10.5, 1.6, 69.6, 7.1, 7.2, 4.0),
    "llama-3.3-70b": (.447, .008, 11.113, 11.3, 1.9, 67.5, 7.7, 7.3, 4.4),
    "gemma-4-4b":    (.432, .011, 11.278, 13.4, 1.8, 58.4, 12.4, 8.7, 5.2),
    "med42-70b":     (.420, .010, 11.524, 12.6, 2.7, 60.9, 10.3, 8.0, 5.5),
    "gemma-4-moe":   (.418, .020, 11.573, 13.3, 2.1, 58.2, 12.3, 8.7, 5.4),
    "gemma-4-31b":   (.416, .013, 11.608, 13.4, 1.8, 59.4, 12.0, 8.0, 5.4),
    "medgemma-27b":  (.415, .015, 11.629, 14.3, 3.1, 55.0, 12.7, 8.8, 6.1),
}

PAPER = {}
for m, (e, s, dpg, esc, att, omi, fab, sco, sev) in T1.items():
    PAPER.update({f"T1.{m}.efs": ("dp", e, 3, "Table 1"), f"T1.{m}.efs_std": ("dp", s, 3, "Table 1"),
                  f"T1.{m}.dist_per_gen": ("dp", dpg, 3, "Table 1"),
                  f"T1.{m}.escalation_pct": ("dp", esc, 1, "Table 1"), f"T1.{m}.attenuation_pct": ("dp", att, 1, "Table 1"),
                  f"T1.{m}.omission_pct": ("dp", omi, 1, "Table 1"), f"T1.{m}.fabrication_pct": ("dp", fab, 1, "Table 1"),
                  f"T1.{m}.scope_shift_pct": ("dp", sco, 1, "Table 1"), f"T1.{m}.severity_warp_pct": ("dp", sev, 1, "Table 1")})

PAPER.update({
    # Section 6.1 / 6.2 / Table A18
    "S6.1.mean_efs_3seeds": ("dp", .450, 3, "6.1"), "S6.1.efs_range_min": ("dp", .415, 3, "6.1"),
    "S6.1.efs_range_max": ("dp", .493, 3, "6.1"), "S6.1.crossseed_std_min": ("dp", .001, 3, "6.1"),
    "S6.1.crossseed_std_max": ("dp", .020, 3, "6.1"), "S6.1.crossseed_std_mean": ("dp", .009, 3, "6.1"),
    "S6.1.med_mean": ("dp", .457, 3, "6.1"), "S6.1.gen_mean": ("dp", .445, 3, "6.1"),
    "A18.omission.count": ("int", 135412, "Table A18"), "A18.omission.pct": ("dp", 66.3, 1, "Table A18"), "A18.omission.per_gen": ("dp", 7.327, 3, "Table A18"),
    "A18.escalation.count": ("int", 23979, "Table A18"), "A18.escalation.pct": ("dp", 11.7, 1, "Table A18"), "A18.escalation.per_gen": ("dp", 1.298, 3, "Table A18"),
    "A18.fabrication.count": ("int", 17205, "Table A18"), "A18.fabrication.pct": ("dp", 8.4, 1, "Table A18"), "A18.fabrication.per_gen": ("dp", .931, 3, "Table A18"),
    "A18.scope_shift.count": ("int", 14397, "Table A18"), "A18.scope_shift.pct": ("dp", 7.0, 1, "Table A18"), "A18.scope_shift.per_gen": ("dp", .779, 3, "Table A18"),
    "A18.severity_warp.count": ("int", 9548, "Table A18"), "A18.severity_warp.pct": ("dp", 4.7, 1, "Table A18"), "A18.severity_warp.per_gen": ("dp", .517, 3, "Table A18"),
    "A18.attenuation.count": ("int", 3679, "Table A18"), "A18.attenuation.pct": ("dp", 1.8, 1, "Table A18"), "A18.attenuation.per_gen": ("dp", .199, 3, "Table A18"),
    "A18.total": ("int", 204220, "Table A18"), "A18.total_per_gen": ("dp", 11.051, 3, "Table A18"),
    # Table A1
    "A1.model_H": ("dp", 409.8, 1, "Table A1"), "A1.model_p": ("lt", 1e-81, "Table A1"),
    "A1.task_H": ("dp", 113.5, 1, "Table A1"), "A1.task_p": ("lt", 1e-24, "Table A1"),
    "A1.specialty_H": ("dp", 966.9, 1, "Table A1"), "A1.specialty_p": ("lt", 1e-208, "Table A1"),
    "A1.scale_gen_r": ("dp", -.044, 3, "Table A1"), "A1.scale_gen_p": ("lt", 1e-8, "Table A1"),
    "A1.scale_model_r": ("dp", -.281, 3, "Table A1 / 6.1"), "A1.scale_model_p": ("dp", .402, 3, "Table A1 / 6.1"),
    "A1.med_U": ("sci", 4.4e7, 2, "Table A1"), "A1.med_p": ("sci", 3e-4, 1, "Table A1 / 6.1"),
    "A1.domain_H": ("dp", 135.2, 1, "Table A1 / 6.5"), "A1.domain_p": ("sci", 4.4e-30, 2, "6.5"),
    # Appendix B cluster-level
    "B.friedman_blocks_n": ("int", 1680, "App. B"), "B.friedman_blocks_chi2": ("dp", 1863.6, 1, "App. B"),
    "B.friedman_episodes_n": ("int", 560, "App. B"), "B.friedman_episodes_chi2": ("dp", 1437.1, 1, "App. B"),
    "B.med_minus_gen": ("dp", .011, 3, "App. B"), "B.med_minus_gen_ci_lo": ("dp", .008, 3, "App. B"),
    "B.med_minus_gen_ci_hi": ("dp", .014, 3, "App. B"), "B.med_minus_gen_wilcoxon_p": ("sci", 1.1e-16, 2, "App. B"),
    "B.n_episodes": ("int", 560, "App. B"),
    "B.domain_doc_H": ("dp", 28.9, 1, "App. B"), "B.domain_doc_p": ("sci", 5.3e-7, 2, "App. B"),
    "B.clin_vs_sci_p": ("dp", .008, 3, "App. B"), "B.clin_vs_legal_p": ("lt", .002, "App. B"), "B.sci_vs_legal_p": ("lt", .002, "App. B"),
    "B.paired.prompt.llama_med42.p": ("ge", .11, "App. B"), "B.paired.fewshot.llama_med42.p": ("ge", .11, "App. B"),
    "B.paired.posthoc.llama_med42.p": ("ge", .11, "App. B"),
    "B.paired.prompt.mistral.p": ("ge", .07, "App. B"), "B.paired.fewshot.mistral.p": ("ge", .07, "App. B"),
    "B.paired.posthoc.mistral.p": ("ge", .07, "App. B"),
    "B.paired.prompt.gemma.p": ("lt", 1e-11, "App. B"), "B.paired.fewshot.gemma.p": ("lt", 1e-11, "App. B"),
    "B.paired.posthoc.gemma.p": ("lt", 1e-11, "App. B"),
    "B.paired.combo.three.delta": ("dp", .009, 3, "App. B"), "B.paired.combo.three.p": ("dp", .001, 3, "App. B"),
    "B.paired.constrained.three.delta": ("dp", -.010, 3, "App. B"), "B.paired.constrained.three.p": ("dp", .02, 2, "App. B"),
    # Table A5
    "A5.baseline.efs": ("dp", .477, 3, "Table A5"), "A5.baseline.efs_sd": ("dp", .217, 3, "Table A5"),
    **{f"A5.{k}.{f}": ("dp", x, d, "Table A5") for k, vals in {
        "prompt": (.474, .216, -.002, -.011, .959), "fewshot": (.471, .216, -.006, -.026, .719),
        "posthoc": (.480, .214, .004, .017, .653), "T0.3": (.517, .170, .040, .189, .041),
        "T0.7": (.508, .201, .032, .147, .057)}.items()
       for f, x, d in zip(("efs", "efs_sd", "delta", "cohen_d", "mwu_p"), vals, (3, 3, 3, 3, 3))},
    # Table A6
    "A6.baseline.efs": ("dp", .478, 3, "Table A6"), "A6.baseline.n": ("int", 1728, "Table A6"),
    **{f"A6.{k}.{f}": (kind, x, *rest) for k, (n, e, dl, p) in {
        "strict": (1728, .469, -.008, .332), "cot": (1728, .478, .001, .631),
        "combo": (1728, .486, .009, .213), "constrained": (1728, .467, -.010, .231)}.items()
       for f, kind, x, rest in (("n", "int", n, ("Table A6",)), ("efs", "dp", e, (3, "Table A6")),
                                 ("delta", "dp", dl, (3, "Table A6")), ("mwu_p", "dp", p, (3, "Table A6")))},
    # Table A7
    "A7.mistral-7b.baseline.efs": ("dp", .480, 3, "Table A7"), "A7.gemma-4-4b.baseline.efs": ("dp", .420, 3, "Table A7"),
    **{f"A7.{m}.{s}.{f}": ("dp", x, 3, "Table A7") for (m, s), vals in {
        ("mistral-7b", "prompt"): (.475, -.005, -.023, .780), ("mistral-7b", "fewshot"): (.488, .008, .038, .491),
        ("mistral-7b", "posthoc"): (.484, .004, .019, .862), ("gemma-4-4b", "prompt"): (.480, .061, .278, None),
        ("gemma-4-4b", "fewshot"): (.501, .082, .379, None), ("gemma-4-4b", "posthoc"): (.461, .042, .190, .002)}.items()
       for f, x in zip(("efs", "delta", "cohen_d", "mwu_p"), vals) if x is not None},
    "A7.gemma-4-4b.prompt.mwu_p": ("lt", 1e-5, "Table A7"), "A7.gemma-4-4b.fewshot.mwu_p": ("lt", 1e-8, "Table A7"),
    # Cross-domain, Figure A1, Appendix E / J.3 / Q, F.1
    "X.clinical3.efs": ("dp", .478, 3, "6.5"), "X.legal.efs": ("dp", .359, 3, "6.5"), "X.scientific.efs": ("dp", .427, 3, "6.5"),
    "X.clinical.efs": ("dp", .443, 3, "Fig. A1"), "X.clinical.efs_sd": ("dp", .220, 3, "Fig. A1"),
    "X.legal.efs_sd": ("dp", .337, 3, "Fig. A1"), "X.scientific.efs_sd": ("dp", .257, 3, "Fig. A1"),
    "X.clinical.pct_floor": ("dp", 9.0, 1, "J.3"), "X.scientific.pct_floor": ("dp", 12.8, 1, "J.3"), "X.legal.pct_floor": ("dp", 34.7, 1, "J.3"),
    "X.clinical.pct_clamp_binds": ("dp", 4.8, 1, "J.3"), "X.scientific.pct_clamp_binds": ("dp", 5.2, 1, "J.3"),
    "X.legal.pct_clamp_binds": ("dp", 20.7, 1, "J.3"),
    "X.clinical.unclamped": ("dp", .422, 3, "J.3"), "X.scientific.unclamped": ("dp", .411, 3, "J.3"), "X.legal.unclamped": ("dp", .209, 3, "J.3"),
    "X.clinical.rank_rho_unclamped": ("dp", .964, 3, "J.3"), "X.legal.rank_rho_unclamped": ("dp", 1.0, 3, "J.3"),
    "X.scientific.rank_rho_unclamped": ("dp", 1.0, 3, "J.3"),
    "X.clinical.no_omission": ("dp", .747, 3, "App. E"), "X.legal.no_omission": ("dp", .574, 3, "App. E"),
    "X.scientific.no_omission": ("dp", .620, 3, "App. E"),
    "X.clinical.any_escalation_pct": ("dp", 57, 0, "App. E"), "X.legal.any_escalation_pct": ("dp", 38, 0, "App. E"),
    "X.scientific.any_escalation_pct": ("dp", 70, 0, "App. E"),
    "X.legal.core4": ("dp", .392, 3, "App. Q"), "X.clinical3.core4": ("dp", .541, 3, "App. Q"), "X.clinical.core4": ("dp", .510, 3, "App. Q"),
    "X.clinical.efs_w": ("dp", .531, 3, "F.1"), "X.legal.efs_w": ("dp", .378, 3, "F.1"), "X.scientific.efs_w": ("dp", .411, 3, "F.1"),
    "X.clinical.r_efs_efsw": ("dp", .874, 3, "F.1"), "X.legal.r_efs_efsw": ("dp", .893, 3, "F.1"),
    "X.clinical.rank_rho_efsw": ("dp", .982, 3, "F.1"), "X.legal.rank_rho_efsw": ("dp", 1.0, 3, "F.1"),
    "X.scientific.rank_rho_efsw": ("dp", 1.0, 3, "F.1"),
    "F1.pooled_n": ("int", 8640, "F.1"), "F1.pooled_r": ("dp", .875, 3, "F.1"), "F1.per_model_r": ("dp", .992, 3, "F.1"),
    # Table 2 / A2 / Appendix E (hosted)
    "T2.n_scored": ("int", 7005, "Table 2"),
    **{f"T2.{m}.{d}{suf}": ("dp", x, 3, "Table 2") for m, vals in {
        "Phi-4": (.416, .045, .406, .057, .483, .006), "gpt-4o": (.342, .037, .373, .045, .468, .016),
        "Mistral-Large-3": (.302, .034, .335, .056, .443, .006), "gpt-5.4": (.201, .022, .343, .060, .422, .006)}.items()
       for (d, suf), x in zip((("legal", ""), ("legal", ".sd"), ("medical", ""), ("medical", ".sd"),
                               ("scientific", ""), ("scientific", ".sd")), vals)},
    # printed in scientific notation when the paper does (x < .001, and 9.7e-3), else as 3 decimals
    **{f"A2.{d}.{p}": (("sci", x, 2) if (x < .001 or x == 9.7e-3) else ("dp", x, 3)) + ("Table A2",) for d, pv in {
        "legal": {"Phi-4|gpt-4o": 2.5e-4, "Phi-4|Mistral-Large-3": 4.0e-9, "Phi-4|gpt-5.4": 2.6e-30,
                  "gpt-4o|Mistral-Large-3": .014, "gpt-4o|gpt-5.4": 4.2e-17, "Mistral-Large-3|gpt-5.4": 5.8e-9},
        "medical": {"Phi-4|gpt-4o": .146, "Phi-4|Mistral-Large-3": 5.8e-4, "Phi-4|gpt-5.4": .003,
                    "gpt-4o|Mistral-Large-3": .044, "gpt-4o|gpt-5.4": .112, "Mistral-Large-3|gpt-5.4": .701},
        "scientific": {"Phi-4|gpt-4o": .465, "Phi-4|Mistral-Large-3": 9.7e-3, "Phi-4|gpt-5.4": 4.8e-5,
                       "gpt-4o|Mistral-Large-3": .068, "gpt-4o|gpt-5.4": .001, "Mistral-Large-3|gpt-5.4": .151}}.items()
       for p, x in pv.items()},
    "E.legal_capped.Mistral-Large-3": ("int", 414, "App. E"), "E.legal_n.Mistral-Large-3": ("int", 592, "App. E"),
    "E.legal_capped.gpt-5.4": ("int", 196, "App. E"), "E.legal_n.gpt-5.4": ("int", 590, "App. E"),
    "E.legal_capped.gpt-4o": ("int", 73, "App. E"), "E.legal_n.gpt-4o": ("int", 588, "App. E"),
    "E.legal_capped.Phi-4": ("int", 0, "App. E"),
    "E.legal_capped_efs.gpt-5.4": ("dp", .100, 3, "App. E"), "E.legal_uncapped_efs.gpt-5.4": ("dp", .251, 3, "App. E"),
    "E.legal_capped_efs.gpt-4o": ("dp", .267, 3, "App. E"), "E.legal_uncapped_efs.gpt-4o": ("dp", .353, 3, "App. E"),
    "E.mean_completion_tokens.Phi-4": ("dp", 348, 0, "App. E"), "E.mean_completion_tokens.Mistral-Large-3": ("dp", 456, 0, "App. E"),
    "E.mean_completion_tokens.gpt-5.4": ("dp", 399, 0, "App. E"),
    "E.band_n": ("int", 2937, "App. E"), "E.band_phi": ("dp", .487, 3, "App. E"), "E.band_others": ("dp", .396, 3, "App. E"),
    "E.band_U": ("dp", 979855, 0, "App. E"), "E.band_p": ("sci", 1.6e-11, 2, "App. E"),
    "E.hosted_ols_n": ("int", 6971, "App. E"), "E.hosted_ols_coef": ("dp", -.076, 3, "App. E"),
    "E.hosted_ols_ci_lo": ("dp", -.103, 3, "App. E"), "E.hosted_ols_ci_hi": ("dp", -.049, 3, "App. E"),
    "E.hosted_ols_r2": ("dp", .056, 3, "App. E"), "E.hosted_loglen_r2": ("dp", .026, 3, "App. E"),
    # Table A4, Appendix C / E (length)
    "C.depth_over_1024": ("int", 1004, "App. C"), "C.depth_at_2048_cap": ("int", 135, "App. C"),
    "C.legal_over_1024": ("int", 73, "App. C"), "C.scientific_over_1024": ("int", 5, "App. C"),
    **{f"A4.{q}.{f}": (("int", x) if f == "n" else ("dp", x, 0 if f == "mean_words" else 3)) + ("Table A4",)
       for q, vals in {"Q1": (1272, 224, .377, .473), "Q2": (1274, 339, .320, .441), "Q3": (1256, 390, .313, .418),
                       "Q4": (1273, 441, .292, .425), "Q5": (1261, 581, .290, .455)}.items()
       for f, x in zip(("n", "mean_words", "omission_per_hedge", "efs"), vals)},
    "E.Q5_om_min": ("dp", .245, 3, "App. E"), "E.Q5_om_max": ("dp", .332, 3, "App. E"),
    "E.len_ols_coef": ("dp", -.060, 3, "App. E"), "E.len_ols_ci_lo": ("dp", -.074, 3, "App. E"),
    "E.len_ols_ci_hi": ("dp", -.045, 3, "App. E"), "E.len_ols_p": ("sci", 9e-16, 1, "App. E"), "E.len_ols_r2": ("dp", .035, 3, "App. E"),
})

# Printed values that differ from what this code gives, with the reason.
KNOWN_DIFFERENCES = {}

# Appendix D, J.1, J.4, J.5, J.6, Table A11 (scripts/analysis/aligner_and_markers.py)
PAPER.update({
    "J5.greedy_depth_efs": ("dp", .416, 3, "J.5"), "J5.greedy_omission_pct": ("dp", 43.5, 1, "J.5"),
    "J5.hungarian_omission_pct": ("dp", 66.3, 1, "J.5"), "J5.rank_spearman_open": ("dp", .936, 3, "J.5"),
    "J5.same_top_bottom": ("int", 1, "J.5"), "J5.rank_spearman_hosted": ("dp", 1.0, 3, "J.5"),
    "J5.identity_combinations": ("int", 1728, "J.5"), "J5.identity_all_zero": ("int", 1, "J.5"),
    "A11.clinical.identity_efs": ("dp", .864, 3, "Table A11"), "A11.legal.identity_efs": ("dp", .872, 3, "Table A11"),
    "A11.scientific.identity_efs": ("dp", .788, 3, "Table A11"),
    "A11.clinical.12000.greedy_fpr_pct": ("dp", 66.7, 1, "Table A11"), "A11.legal.12000.greedy_fpr_pct": ("dp", 43.2, 1, "Table A11"),
    "A11.scientific.12000.greedy_fpr_pct": ("dp", 69.8, 1, "Table A11"),
    "A11.clinical.observed_greedy": ("dp", .414, 3, "Table A11"), "A11.legal.observed_greedy": ("dp", .361, 3, "Table A11"),
    "A11.scientific.observed_greedy": ("dp", .395, 3, "Table A11"),
    "A11.clinical.3000.greedy_fpr_pct": ("dp", 18.8, 1, "J.5"), "A11.clinical.6000.greedy_fpr_pct": ("dp", 50.5, 1, "J.5"),
    "A11.clinical.12000.greedy_spurious": ("int", 452, "J.5"),
    "J6.floor_pct.omission": ("dp", 0.0, 1, "J.6"), "J6.floor_pct.fabrication": ("dp", 0.0, 1, "J.6"),
    "J6.floor_pct.attenuation": ("dp", 6.7, 1, "J.6"), "J6.floor_pct.scope_shift": ("dp", 12.9, 1, "J.6"),
    "J6.floor_pct.severity_warp": ("dp", 31.6, 1, "J.6"), "J6.floor_pct.escalation": ("dp", 51.6, 1, "J.6"),
    "J6.omission_plus_fabrication_pct": ("dp", 74.7, 1, "J.6"),
    "J4.markers_per_doc.seed123": ("dp", 15.8, 1, "App. D / J.4"), "J4.markers_per_doc.seed0": ("dp", 23.5, 1, "J.4"),
    "J4.markers_per_doc.seed1": ("dp", 22.4, 1, "J.4"), "J4.markers_per_doc.3seeds": ("dp", 20.5, 1, "J.4"),
    "D.contextual_per_doc": ("dp", 6.0, 1, "App. D"), "D.contextual_reduction_pct": ("dp", 62, 0, "App. D"),
    "J4.severity_temporal_pct": ("dp", 74.7, 1, "J.1 / J.4"), "J1.strength_zero_pct": ("dp", 84.3, 1, "J.1"),
    "J4.scientific_quantitative_pct": ("dp", 44, 0, "J.4"),
    "J5.sweep.efs_min": ("dp", .441, 3, "6.1 / J.5"), "J5.sweep.efs_max": ("dp", .449, 3, "6.1 / J.5"),
    "J5.sweep.0.30.efs": ("dp", .442, 3, "J.5"), "J5.sweep.0.20.omission_pct": ("dp", 56.3, 1, "J.5"),
    "J5.sweep.0.30.omission_pct": ("dp", 61.2, 1, "J.5"), "J5.sweep.0.50.omission_pct": ("dp", 74.3, 1, "J.5"),
})


def _dp(x, k, where):
    return ("dp", x, k, where)


# Controlled ground truth (scripts/validation/controlled_groundtruth_validation.py)
PAPER.update({
    "J7.rho_mean": _dp(-.891, 3, "J.7 / Table A12"), "J7.rho_std": _dp(.006, 3, "J.5"),
    **{f"J7.mean_efs.{k}": _dp(x, 3, "Table A12") for k, x in
       (("0.00", 1.0), ("0.25", .830), ("0.50", .644), ("0.75", .443), ("1.00", .249))},
    "J5.gt_greedy.rho_mean": _dp(-.721, 3, "J.5"), "J5.gt_greedy.rho_std": _dp(.009, 3, "J.5"),
    "J7.claude.severity_vs_fraction_rho": _dp(.650, 3, "J.7"),
})

# Restricted markers J.4, properties J.3, Table A10, F.2, Section 6.3 (scripts/analysis, scripts/validation)
PAPER.update({
    "J4.clinical_3seed_efs": _dp(.366, 3, "6.1 / J.4"), "J4.markers_per_doc": _dp(6.2, 1, "J.4"),
    "J4.rank_spearman_vs_unrestricted": _dp(.873, 3, "6.1 / J.4"), "J4.rank_spearman_p": ("lt", 1e-3, "J.4"),
    "J4.rank.biomistral-7b": ("int", 1, "J.4"), "J4.rank.medgemma-27b": ("int", 11, "J.4"),
    "J4.omission_pct": _dp(60.3, 1, "J.4"), "J4.escalation_pct": _dp(21.3, 1, "6.2 / J.4"),
    "J4.unrestricted.crossseed_std_max": _dp(.020, 3, "J.4"), "J4.crossseed_std_max": _dp(.067, 3, "J.4"),
    "J4.matched.clinical_efs": _dp(.456, 3, "J.4"), "J4.matched.scientific_efs": _dp(.415, 3, "J.4"),
    "J4.matched.legal_efs": _dp(.359, 3, "J.4"), "J4.matched.kruskal_H": _dp(44.8, 1, "J.4"),
    "J4.matched.kruskal_p": ("sci", 1.9e-10, 2, "J.4"),
    "J3.n_notes": ("int", 60, "J.3"), "J3.markers_per_note": _dp(16.9, 1, "J.3"),
    "J3.identity_notes_with_distortion": ("int", 0, "J.3"), "J3.greedy_identity_mean_efs": _dp(.823, 3, "J.3"),
    "J3.greedy_identity_fpr_pct": _dp(80.0, 1, "J.3"), "J3.deletion_steps": ("int", 919, "J.3"),
    "J3.deletion_violations": ("int", 0, "J.3"), "J3.efs_after_all_deletions": _dp(.340, 3, "J.3"),
    "J3.scale_max_abs_delta": _dp(0, 3, "J.3"), "J3.graft_notes_decreased": ("int", 59, "J.3"),
    "J3.graft_efs_after": _dp(.907, 3, "J.3"), "J3.certain_notes_decreased": ("int", 1, "J.3"),
    "J3.markers_never_in_claim_pct": _dp(39, 0, "J.3"), "J3.efs0.implicated_pct_pooled": _dp(68, 0, "J.3"),
    "A10.markers_per_note": _dp(19.1, 1, "J.3"), "A10.notes_shared_with_J3": ("int", 53, "J.3"),
    **{f"A10.{row}.{col}": _dp(x, 3, "Table A10") for row, vals in {
        "identity": (1.0, 1.0, 1.0), "reorder": (.998, 1.0, .334), "delete_0.50": (.670, .498, .996),
        "relocate": (.806, 1.0, .997), "add_hedge": (.935, 1.218, .998), "fabricate": (.931, 1.070, .993)}.items()
       for col, x in zip(("efs", "hedge_count_ratio", "rougeL"), vals)},
    "A10.deletion_spearman.efs": _dp(-.897, 3, "J.3"), "A10.deletion_spearman.hedge_count_ratio": _dp(-.988, 3, "J.3"),
    "A10.deletion_spearman.rougeL": _dp(-.864, 3, "J.3"),
    "F2.identity.efs": _dp(1.0, 3, "F.2"), "F2.dehedge.efs": _dp(.268, 3, "6.3 / F.2"), "F2.dehedge.rouge_l": _dp(.994, 3, "6.3 / F.2"),
    "F2.content_cut.efs": _dp(.757, 3, "F.2"), "F2.content_cut.rouge_l": _dp(.676, 3, "F.2"),
    "F2.dehedge.greedy_efs": _dp(.270, 3, "J.5"),
    "S6.3.n": ("int", 6336, "6.3"), "S6.3.rouge_r": _dp(.107, 3, "6.3"), "S6.3.rouge_p": ("sci", 1.2e-17, 2, "6.3"),
    "S6.3.rouge_r2": _dp(.011, 3, "6.3"), "S6.3.bertscore_r": _dp(-.101, 3, "6.3"), "S6.3.bertscore_p": ("sci", 6.5e-16, 2, "6.3"),
})

# F.3-F.6 (scripts/analysis, scripts/decision)
PAPER.update({
    "F3.n": ("int", 200, "F.3"), "F3.efs_before": _dp(.493, 3, "F.3"), "F3.efs_after": _dp(.832, 3, "F.3"),
    "F3.dist_reduction_pct": _dp(66, 0, "F.3"), "F3.greedy_efs_before": _dp(.458, 3, "F.3"),
    "F3.greedy_efs_after": _dp(.732, 3, "F.3"), "F3.greedy_dist_reduction_pct": _dp(50, 0, "F.3"),
    "F4.legal.coef_is_model": _dp(.025, 3, "F.4"), "F4.legal.coef_p": _dp(.057, 3, "F.4"),
    "F4.legal.coef_ci_lo": _dp(-.001, 3, "F.4"), "F4.legal.coef_ci_hi": _dp(.050, 3, "F.4"),
    "F4.legal.band_n_human": ("int", 180, "F.4"), "F4.legal.band_n_model": ("int", 1152, "F.4"),
    "F4.legal.band_human_harden": _dp(.082, 3, "F.4"), "F4.legal.band_model_harden": _dp(.107, 3, "F.4"),
    "F4.legal.band_mannwhitney_p": _dp(.024, 3, "F.4"), "F4.scientific.coef_is_model": _dp(.117, 3, "F.4"),
    "F4.scientific.coef_p": ("sci", 1.6e-11, 2, "F.4"), "F4.scientific.human_compression": _dp(.021, 3, "F.4"),
    "F4.scientific.model_compression": _dp(.317, 3, "F.4"), "F4.scientific.band_n_human": ("int", 4, "F.4"),
    "F5.proxy.n": ("int", 150, "F.5"), "F5.proxy.any_change_pct": _dp(82.7, 1, "F.5"),
    "F5.proxy.urgency_changed_pct": _dp(65, 0, "F.5"), "F5.proxy.action_changed_pct": _dp(46, 0, "F.5"),
    "F5.identity_floor.any_change_pct": _dp(27.3, 1, "F.5 / J.8"), "F5.paired.any.mcnemar_exact_p": ("sci", 1.1e-20, 2, "F.5"),
    "F5.harm.n": ("int", 150, "F.5"), "F5.harm.pointbiserial_r": _dp(.010, 3, "F.5"), "F5.harm.pointbiserial_p": _dp(.908, 3, "F.5"),
    **{f"F5.harm.bin{i}.change_pct": _dp(x, 1, "F.5") for i, x in enumerate((70.8, 75.0, 82.1, 73.1, 75.0), 1)},
    "F6.n": ("int", 185, "F.6"), "F6.dehedge_change_pct": _dp(15.7, 1, "F.6"), "F6.identity_floor_pct": _dp(1.1, 1, "F.6"),
    "F6.excess_pts": _dp(14.6, 1, "F.6"), "F6.bootstrap_ci_lo": _dp(9.2, 1, "F.6"), "F6.bootstrap_ci_hi": _dp(20.0, 1, "F.6"),
    "F6.mcnemar_b_dehedge_only": ("int", 28, "F.6"), "F6.mcnemar_c_identity_only": ("int", 1, "F.6"),
    "F6.mcnemar_exact_p": ("sci", 1.1e-7, 2, "F.6"), "F6.left_padded_identity_floor_pct": _dp(12.5, 1, "F.6"),
})

# Appendix I / Table A8 / Section 6.6 / J.8 (scripts/judges)
PAPER.update({
    "I.positives.n": ("int", 242, "App. I"), "I.positives.per_model_min": ("int", 22, "App. I"),
    "I.positives.per_model_max": ("int", 22, "App. I"), "I.positives.judge70b_flagged": ("int", 242, "App. I"),
    "I.positives.pearson_r": _dp(.186, 3, "App. I / Table A8"), "I.positives.pearson_p": _dp(.004, 3, "App. I / Table A8"),
    "A8.truncation.judge70b_flagged": ("int", 139, "Table A8"), "I.truncation.judge70b_flag_pct": _dp(57.4, 1, "App. I"),
    "I.identity.judge70b.flagged": ("int", 1, "6.6 / App. I"), "I.identity.judge70b.flag_pct": _dp(0.4, 1, "App. I"),
    "I.identity.judge70b.wilson_lo_pct": _dp(0.1, 1, "App. I"), "I.identity.judge70b.wilson_hi_pct": _dp(2.3, 1, "App. I"),
    "I.identity.judge70b.parse_fail": ("int", 0, "App. I"), "I.identity.judge70b.rated_same": ("int", 242, "App. I"),
    "I.identity.claude.flag_pct": _dp(0.0, 1, "App. I"), "I.identity.claude.wilson_hi_pct": _dp(1.6, 1, "App. I"),
    "I.truncation.mean_source_chars": _dp(10210, 0, "App. I"), "I.truncation.hedges_retained_pct": _dp(24.7, 1, "App. I"),
    "I.truncation.pairs_losing_hedge": ("int", 236, "App. I"), "A8.truncation.pairs_losing_hedge_pct": _dp(97.5, 1, "Table A8"),
    "A8.truncation.pipeline_greedy_flag_pct": _dp(89.7, 1, "App. I / Table A8"),
    "A8.identity.pipeline_greedy_fpr_pct": _dp(27.7, 1, "App. I / Table A8"),
    "I.positive_control.P.flag_pct": _dp(98.3, 1, "App. I"), "I.positive_control.P.mean_severity": _dp(2.833, 3, "App. I"),
    "I.positive_control.P.distinct_explanations": ("int", 60, "App. I"), "I.positive_control.D.flag_pct": _dp(60.0, 1, "App. I"),
    "I.positive_control.T.flag_pct": _dp(1.7, 1, "App. I"), "I.positive_control.I.flag_pct": _dp(0.0, 1, "App. I"),
    "S6.6.n_submitted": ("int", 750, "6.6"), "S6.6.strict.n": ("int", 541, "6.6"),
    "S6.6.strict.more_certain_pct": _dp(96.3, 1, "6.6"), "S6.6.strict.clinical.more_certain_pct": _dp(98.3, 1, "6.6"),
    "S6.6.strict.scientific.more_certain_pct": _dp(94.6, 1, "6.6"), "S6.6.strict.less_certain_pct": _dp(2.8, 1, "6.6"),
    "J8.recovered.more_certain_pct": _dp(97.1, 1, "6.6 / J.8"), "J8.recovered.clinical.more_certain_pct": _dp(98.7, 1, "J.8"),
    "J8.recovered.scientific.more_certain_pct": _dp(94.7, 1, "J.8"), "J8.recovered.less_certain_pct": _dp(2.0, 1, "J.8"),
    "J8.code_fenced_recovered": ("int", 209, "J.8"),
})

# RadGraph (6.7, J.1, J.2, Table A9), NUBes (J.1), annotators (Table A13, App. K)
PAPER.update({
    "J2.control.n": ("int", 389, "6.7"), "J2.control.n_reports": ("int", 161, "J.2"),
    "J2.control.source_preserved_pct": _dp(83.8, 1, "6.7"), "J2.control.model_preserved_pct": _dp(54.5, 1, "6.7"),
    "J2.control.drop_pts": _dp(29.3, 1, "6.7"), "J2.control.drop_p": ("sci", 1.2e-25, 2, "6.7"),
    "J2.control.drop_ci_lo": _dp(23.3, 1, "6.7"), "J2.control.drop_ci_hi": _dp(35.3, 1, "6.7"),
    "J2.control.omitted_pct": _dp(3.9, 1, "J.2"), "J2.both_mentioned.n": ("int", 329, "6.7"),
    "J2.both_mentioned.source_pct": _dp(14.3, 1, "6.7"), "J2.both_mentioned.model_pct": _dp(36.5, 1, "6.7"),
    "J2.both_mentioned.p": ("sci", 2.5e-16, 2, "J.2"), "J2.survival": _dp(.510, 3, "6.7 / J.2"),
    "J2.survival_ci_lo": _dp(.465, 3, "6.7"), "J2.survival_ci_hi": _dp(.555, 3, "6.7"),
    "J2.survival.biomistral-7b": _dp(.512, 3, "J.2"), "J2.survival.llama-3.1-8b": _dp(.540, 3, "J.2"),
    "J2.survival.med42-8b": _dp(.513, 3, "J.2"), "J2.survival.mistral-7b": _dp(.476, 3, "J.2"),
    "J2.n_judgements": ("int", 2720, "J.2"), "J2.paired_drop_p_max": ("lt", 2.00001e-4, "J.2"),
    "J2.mean_chars.source": _dp(795, 0, "J.2"), "J2.mean_chars.model": _dp(367, 0, "J.2"),
    **{f"A9.{q}.{s}.{f}": ((("int", x) if f == "n" else ("sci", x, 2) if f == "p" else ("dp", x, 1)) + ("Table A9",))
       for (q, s), vals in {
           ("preserved", "primary"): (389, 83.8, 54.5, -29.3, 1.2e-25, -35.3, -23.3),
           ("preserved", "secondary"): (327, 89.0, 57.5, -31.5, 6.4e-25, -38.0, -25.0),
           ("escalated", "primary"): (389, 12.3, 30.8, 18.5, 1.3e-15, 13.4, 23.5),
           ("escalated", "secondary"): (327, 6.4, 25.4, 19.0, 6.4e-15, 13.8, 24.2)}.items()
       for f, x in zip(("n", "source_pct", "model_pct", "delta_pts", "p", "ci_lo", "ci_hi"), vals)},
    "A9.hardened_mentioned.primary.n_source": ("int", 374, "Table A9"), "A9.hardened_mentioned.primary.n_model": ("int", 332, "Table A9"),
    "A9.hardened_mentioned.primary.source_pct": _dp(12.8, 1, "Table A9"), "A9.hardened_mentioned.primary.model_pct": _dp(36.1, 1, "Table A9"),
    "A9.hardened_mentioned.primary.delta_pts": _dp(23.3, 1, "Table A9"), "A9.hardened_mentioned.primary.ratio": _dp(2.816, 3, "6.7 / Table A9"),
    "A9.hardened_mentioned.primary.ratio_ci_lo": _dp(2.130, 3, "Table A9"), "A9.hardened_mentioned.primary.ratio_ci_hi": _dp(3.954, 3, "Table A9"),
    "A9.hardened_mentioned.primary.delta_ci_lo": _dp(17.5, 1, "Table A9"), "A9.hardened_mentioned.primary.delta_ci_hi": _dp(29.2, 1, "Table A9"),
    "J1.radgraph.pooled.epistemic.n": ("int", 3622, "6.7 / J.1"), "J1.radgraph.pooled.epistemic.precision": _dp(.570, 3, "6.7 / J.1"),
    "J1.radgraph.pooled.epistemic.recall": _dp(.609, 3, "6.7 / J.1"), "J1.radgraph.pooled.epistemic.f1": _dp(.588, 3, "6.7 / J.1"),
    "J1.radgraph.test_r1.epistemic.f1": _dp(.606, 3, "J.1"), "J1.radgraph.test_r2.epistemic.f1": _dp(.691, 3, "J.1"),
    "J1.radgraph.pooled.all_spans.precision": _dp(.170, 3, "J.1"), "J1.radgraph.interrater.n_sentences": ("int", 1011, "J.1"),
    "J1.radgraph.interrater.agreement": _dp(.989, 3, "J.1"), "J1.radgraph.interrater.kappa": _dp(.879, 3, "J.1"),
    "J1.radgraph.interrater.f1": _dp(.884, 3, "6.7 / J.1"),
    "J1.nubes.n_files": ("int", 615, "J.1"), "J1.nubes.n_cues": ("int", 2795, "J.1"), "J1.nubes.n_forms": ("int", 321, "J.1"),
    "J1.nubes.handbuilt.cue_recall_mean": _dp(.448, 3, "J.1"), "J1.nubes.handbuilt.cue_recall_std": _dp(.019, 3, "J.1"),
    "J1.nubes.handbuilt.cue_recall_min": _dp(.420, 3, "J.1"), "J1.nubes.handbuilt.precision_mean": _dp(.824, 3, "J.1"),
    "J1.nubes.handbuilt.recall_mean": _dp(.487, 3, "J.1"), "J1.nubes.handbuilt.f1_mean": _dp(.612, 3, "6.7 / J.1"),
    "J1.nubes.derived100.cue_recall_mean": _dp(.909, 3, "J.1"), "J1.nubes.derived100.cue_recall_std": _dp(.008, 3, "J.1"),
    "J1.nubes.derived100.precision_mean": _dp(.250, 3, "J.1"), "J1.nubes.derived100.f1_mean": _dp(.394, 3, "J.1"),
    **{f"A13.vs_human.{a}.ac1": _dp(x, 3, "Table A13") for a, x in
       (("claude-sonnet-4-6", .947), ("llama-3.3-70b", .942), ("gpt-oss-20b", .934), ("gemma-4-31b", .559))},
    "A13.vs_human.gpt-oss-20b.n": ("int", 18, "Table A13"),
    **{f"A13.vs_pipeline.{a}.ac1": _dp(x, 3, "Table A13") for a, x in
       (("claude-sonnet-4-6", .907), ("llama-3.3-70b", .921), ("gpt-oss-20b", .792), ("gemma-4-31b", .607))},
    "A13.vs_pipeline.gpt-oss-20b.n": ("int", 139, "Table A13"), "A13.vs_pipeline.claude-sonnet-4-6.n": ("int", 164, "Table A13"),
    "A13.ai_vs_ai.claude-sonnet-4-6|llama-3.3-70b.ac1": _dp(.948, 3, "J.9"),
    "K.severity_qwk.mean": _dp(.430, 3, "App. K"), "K.severity_qwk.R1-R2": _dp(.554, 3, "App. K"),
    "K.severity_qwk.R1-R3": _dp(.543, 3, "App. K"), "K.severity_qwk.R2-R3": _dp(.194, 3, "App. K"),
    "K.severity_within1_pct": _dp(70, 0, "App. K"), "K.fleiss.fabrication": _dp(.603, 3, "App. K"),
    "K.positives.fabrication.R1": ("int", 13, "App. K"), "K.positives.fabrication.R2": ("int", 14, "App. K"),
    "K.positives.fabrication.R3": ("int", 15, "App. K"), "K.positives.omission.R1": ("int", 19, "App. K"),
    "K.positives.omission.R2": ("int", 6, "App. K"), "K.any_distortion.raw_pairwise": _dp(.93, 2, "App. K"),
    "K.any_distortion.ac1_mean_pairwise": _dp(.926, 3, "App. K"), "K.pearson_r": _dp(-.034, 3, "App. K"),
    "K.pearson_boot_lo": _dp(-.523, 3, "App. K"), "K.pearson_boot_hi": _dp(.457, 3, "App. K"),
    "K.spearman_rho": _dp(-.070, 3, "App. K"), "K.spearman_boot_lo": _dp(-.554, 3, "App. K"),
    "K.spearman_boot_hi": _dp(.440, 3, "App. K"),
})

# Token probe (Appendix M, Table A15), detector (Tables A14, A16)
PAPER.update({
    **{f"M.{m}.{k}": _dp(x, d, "Table A15") for m, vals in {
        "llama-3.1-8b": (24.818, 3.449, 89.7), "mistral-7b": (83.080, 3.574, 96.9),
        "med42-8b": (28.696, 3.673, 90.3), "gemma-3-4b": (82.334, 7.605, 91.9)}.items()
       for (k, d), x in zip((("baseline_ratio", 3), ("hedge_ratio", 3), ("reduction_pct", 1)), vals)},
    "M.llama-3.1-8b.hedge_frac_assertive_pct": _dp(90.0, 1, "6.8 / M"),
    "M.llama-3.1-8b.baseline_frac_assertive_pct": _dp(99.8, 1, "M"),
    "M.llama-3.1-8b.n_assertive_ids": ("int", 20, "M"), "M.llama-3.1-8b.n_hedge_ids": ("int", 22, "M"),
    "M.length_match_p": _dp(.501, 3, "M"),
    **{f"A14.{enc}.{dom}.{k}": _dp(x, 1, "Table A14") for enc, vals in {
        "PubMedBERT": (49.2, 58.8, 87.6, 55.2, 74.6, 96.6), "RoBERTa-base": (64.5, 71.0, 87.9, 74.9, 87.5, 97.0),
        "RoBERTa-large": (69.9, 73.5, 87.8, 84.0, 93.2, 97.0), "SciBERT": (63.1, 64.5, 88.0, 72.8, 80.6, 96.9)}.items()
       for (dom, k), x in zip((("legal", "acc"), ("scientific", "acc"), ("clinical_val", "acc"), ("legal", "binary_acc"),
                               ("scientific", "binary_acc"), ("clinical_val", "binary_acc")), vals)},
    "A14.RoBERTa-base.legal.n": ("int", 482, "Table A14"), "A14.RoBERTa-base.scientific.n": ("int", 279, "Table A14"),
    **{f"A16.{enc}.{k}": _dp(x, d, "Table A16") for enc, vals in {
        "RoBERTa-base": (88.7, .2, .836, .814, .853, .736, .620), "PubMedBERT": (88.8, .1, .835, .817, .863, .735, .612),
        "SciBERT": (88.8, .1, .838, .817, .864, .738, .615), "RoBERTa-large": (88.5, .1, .833, .812, .848, .727, .621)}.items()
       for (k, d), x in zip((("acc_mean", 1), ("acc_std", 1), ("macro_f1_mean", 3), ("f1_escalation", 3),
                             ("f1_attenuation", 3), ("f1_scope_shift", 3), ("f1_severity_warp", 3)), vals)},
})

# Values that depend on GPU arithmetic, retraining or recomputed third-party metrics: agreement is
# expected to within about 1% (probe, BERTScore) or about a point (retrained detector), not exactly.
APPROXIMATE = {
    "M.": "float16/bfloat16 next-token probabilities differ slightly across GPUs and library versions",
    "A14.": "exact only with the paper's checkpoints; retrained detectors differ",
    "A16.": "exact only with the paper's 12 runs; retraining uses a different train/validation split",
    "S6.3.bertscore_": "recomputed BERTScores can differ in the fourth decimal (library and GPU numerics)",
}

# BioScope (scripts/validation/bioscope_extractor_validation.py), Appendix J.1
PAPER.update({
    "J1.bioscope.clinical.n_sentences": ("int", 11871, "J.1"),
    "J1.bioscope.scientific.f1": _dp(.559, 3, "6.7 / J.1"), "J1.bioscope.scientific.precision": _dp(.628, 3, "J.1"),
    "J1.bioscope.clinical.f1": _dp(.395, 3, "6.7 / J.1"), "J1.bioscope.clinical.precision": _dp(.533, 3, "J.1"),
    "J1.bioscope.clinical.recall": _dp(.313, 3, "J.1"), "J1.bioscope.contextual.f1": _dp(.082, 3, "J.1"),
    "J1.bioscope.contextual.precision": _dp(.405, 3, "J.1"), "J1.bioscope.contextual.recall": _dp(.046, 3, "J.1"),
})

# Multilingual probe (scripts/analysis/multilingual_probe.py), Appendix F.7
PAPER.update({
    "F7.lexicon_patterns.en": ("int", 17, "F.7"), "F7.lexicon_patterns.es": ("int", 19, "F.7"),
    "F7.lexicon_patterns.fr": ("int", 16, "F.7"),
    **{f"F7.{m}.{lang}.retention": _dp(x, 3, "F.7") for m, vals in
       {"gemma-4-4b": (.955, .531, .411), "llama-3.1-8b": (1.541, .562, .434)}.items()
       for lang, x in zip(("en", "es", "fr"), vals)},
})
