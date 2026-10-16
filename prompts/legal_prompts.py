"""
Prompts for legal text generation tasks.

Two tasks:
  A. Case Summary: summarize a court opinion/order
  B. Brief Memo: draft a legal memorandum from case documents
"""

LEGAL_SYSTEM_PROMPT = (
    "You are a legal documentation assistant. Generate accurate, "
    "professional legal text based on the provided source documents. "
    "Ensure all legal details are faithfully represented."
)

LEGAL_TASK_A_PROMPT = """Based on the following court opinion/order, generate a concise case summary covering:

1. **Case Background**: Parties, claims, and procedural posture
2. **Key Legal Issues**: The central legal questions addressed
3. **Court's Analysis**: How the court reasoned through the issues
4. **Holding**: What the court decided
5. **Implications**: Any noted limitations, conditions, or open questions

---

**Source Document:**

{source_text}

---

Generate the case summary now."""


LEGAL_TASK_B_PROMPT = """Based on the following case documents, draft a legal memorandum for a supervising attorney covering:

1. **Issue Statement**: The legal question(s) to be addressed
2. **Brief Answer**: Short preliminary conclusion
3. **Facts**: Relevant facts from the record
4. **Analysis**: Application of law to facts, including strengths and weaknesses
5. **Conclusion**: Recommended course of action

---

**Source Documents:**

{source_text}

---

Generate the legal memorandum now."""


# Mitigation variants
LEGAL_SYSTEM_PROMPT_MITIGATION = (
    "You are a legal documentation assistant. Generate accurate, "
    "professional legal text based on the provided source documents. "
    "Ensure all legal details are faithfully represented. "
    "IMPORTANT: Maintain the exact level of certainty expressed in the "
    "source. If a finding is described as alleged, arguable, likely, "
    "or uncertain, you MUST preserve that qualification in your output. "
    "Do not upgrade allegations to established facts, and do not "
    "remove qualifiers like 'allegedly', 'arguably', or 'may'."
)

LEGAL_SYSTEM_PROMPT_FEWSHOT = (
    "You are a legal documentation assistant. Generate accurate, "
    "professional legal text based on the provided source documents. "
    "Ensure all legal details are faithfully represented.\n\n"
    "CRITICAL: Preserve the exact level of certainty from the source. "
    "Study these examples:\n\n"
    "CORRECT: Source says 'allegedly discriminated' → Output says 'allegedly discriminated'\n"
    "WRONG:   Source says 'allegedly discriminated' → Output says 'discriminated'\n\n"
    "CORRECT: Source says 'the evidence suggests liability' → Output says 'the evidence suggests liability'\n"
    "WRONG:   Source says 'the evidence suggests liability' → Output says 'the defendant is liable'\n\n"
    "CORRECT: Source says 'may constitute a violation' → Output says 'may constitute a violation'\n"
    "WRONG:   Source says 'may constitute a violation' → Output says 'constitutes a violation'\n\n"
    "Every hedge word (alleged, arguably, may, likely, suggests, appears, presumed) "
    "must appear in your output if it appears in the source."
)
