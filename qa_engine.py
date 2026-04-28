import re
import time
from typing import Any, Dict

from transformers import pipeline


MODEL_NAME = "IProject-10/roberta-base-finetuned-squad2"
MAX_CONTEXT_CHARS = 6000
NO_ANSWER_THRESHOLD = 0.15


class QAEngine:
    """Wrapper around a Hugging Face QA pipeline with validation."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
        )

    def _validate_inputs(self, context: str, question: str) -> tuple[str, str]:
        context = (context or "").strip()
        question = (question or "").strip()

        if not context:
            raise ValueError("Context cannot be empty.")
        if not question:
            raise ValueError("Question cannot be empty.")
        if len(context) > MAX_CONTEXT_CHARS:
            raise ValueError(
                f"Context is too long. Please keep it under {MAX_CONTEXT_CHARS} characters."
            )
        return context, question

    def predict(self, context: str, question: str) -> Dict[str, Any]:
        context, question = self._validate_inputs(context, question)

        start_time = time.perf_counter()
        raw = self.qa_pipeline({"question": question, "context": context})
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        answer_text = (raw.get("answer") or "").strip()
        confidence = float(raw.get("score", 0.0))
        start_char = int(raw.get("start", -1))
        end_char = int(raw.get("end", -1))

        no_answer = (not answer_text) or confidence < NO_ANSWER_THRESHOLD
        if no_answer:
            answer_text = "No confident answer found in context."

        return {
            "answer": answer_text,
            "confidence": round(confidence, 4),
            "start": start_char,
            "end": end_char,
            "latency_ms": round(latency_ms, 2),
            "no_answer": no_answer,
        }


def normalize_text(text: str) -> str:
    """Normalize text for metric comparison."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text
