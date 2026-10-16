"""Prompts sent to the four hosted models (Table 2), verbatim from the paper's runs.

These are NOT the prompts of Appendix S, which are the ones used for the open-weight
models. For the hosted models, the legal, scientific and biomedical system prompts ask the
model to preserve qualifiers and uncertainty, and the task prompts are shorter.
Sources are truncated to 12,000 characters before formatting.
"""

LEGAL_SYSTEM = (
    "You are a legal documentation assistant. Summarize legal cases accurately, "
    "preserving the qualifications, conditions, and uncertainty present in the source."
)
LEGAL_TASKS = {
    "case_summary": (
        "Summarize the following case in 3-5 paragraphs covering parties, "
        "claims, holdings, and reasoning.\n\nSource:\n{source}"
    ),
    "legal_memo": (
        "Draft a brief legal memorandum analyzing the central legal issue in "
        "this case, including arguments and the court's reasoning.\n\n"
        "Source:\n{source}"
    ),
}

SCIENTIFIC_SYSTEM = (
    "You are a scientific writing assistant. Summarize research papers accurately, "
    "preserving methodological qualifiers and limits on the strength of conclusions."
)
SCIENTIFIC_TASKS = {
    "paper_summary": (
        "Summarize this paper in 4-6 sentences covering motivation, methods, "
        "results, and limitations.\n\nSource:\n{source}"
    ),
    "plain_language": (
        "Write a plain-language summary of this paper for a general audience "
        "in 3-5 sentences.\n\nSource:\n{source}"
    ),
}

MEDICAL_SYSTEM = (
    "You are a medical research assistant. Summarize biomedical study abstracts "
    "accurately, preserving methodological qualifiers, evidence levels, and limits "
    "on the strength of clinical conclusions."
)
MEDICAL_TASKS = {
    "abstract_summary": (
        "Summarize the following biomedical study in 4-6 sentences covering "
        "the research question, methods, results, and limitations.\n\n"
        "Source:\n{source}"
    ),
    "clinical_takeaway": (
        "Write a 3-5 sentence clinical takeaway from the following study for a "
        "practising clinician, preserving the certainty level of the conclusions.\n\n"
        "Source:\n{source}"
    ),
}

HOSTED_DOMAINS = {
    # domain: (system prompt, task prompts, sample file, id column)
    "legal": (LEGAL_SYSTEM, LEGAL_TASKS, "legal_sample.parquet", "case_id"),
    "scientific": (SCIENTIFIC_SYSTEM, SCIENTIFIC_TASKS, "scientific_sample.parquet", "paper_id"),
    "medical": (MEDICAL_SYSTEM, MEDICAL_TASKS, "medical_sample.parquet", "abstract_id"),
}
