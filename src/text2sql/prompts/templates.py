"""Instruction formatting shared by training and evaluation.

Chat messages (not a single flat string) are the source of truth so both the
SFT trainer and eval scripts can apply the base model's own chat template
rather than hand-rolling special tokens.
"""

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a question, write the "
    "SQLite query that answers the question.\n"
    "Respond with the SQL query only -- no explanation, no markdown code fences."
)


def build_user_message(question: str, schema: str) -> str:
    return f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:"


def build_messages(question: str, schema: str) -> list[dict[str, str]]:
    """Messages for inference: system + user turns only."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, schema)},
    ]


def build_training_messages(question: str, schema: str, query: str) -> list[dict[str, str]]:
    """Messages for SFT: system + user + the gold assistant turn."""
    messages = build_messages(question, schema)
    messages.append({"role": "assistant", "content": query})
    return messages


def _build_question_only_message(question: str) -> str:
    return f"### Question:\n{question}\n\n### SQL:"


def build_few_shot_messages(
    question: str, schema: str, examples: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Messages for the few-shot baseline: system + worked (question, query) pairs + the real question.

    `examples` items are {"question": ..., "query": ...} drawn from the *same*
    database as `question`, so the worked examples don't leak schema the model
    wouldn't otherwise have. The schema is stated once, in the first turn --
    repeating it in every demonstration turn (an earlier version of this
    function did) multiplies prompt length by ~(num_examples + 1) for no
    benefit, since it's the same schema throughout a single conversation.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    remaining = list(examples)
    if remaining:
        first = remaining.pop(0)
        messages.append({"role": "user", "content": build_user_message(first["question"], schema)})
        messages.append({"role": "assistant", "content": first["query"]})
    for ex in remaining:
        messages.append({"role": "user", "content": _build_question_only_message(ex["question"])})
        messages.append({"role": "assistant", "content": ex["query"]})
    messages.append(
        {
            "role": "user",
            "content": build_user_message(question, schema) if not examples else _build_question_only_message(question),
        }
    )
    return messages
