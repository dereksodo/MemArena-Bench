from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LightAnswerRecord:
    question_id: str
    prediction: str
    model: str
    raw_response: str
