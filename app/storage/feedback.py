import json
import uuid
from datetime import datetime, timezone

from app.config import get_settings


def save_feedback(question: str, answer: str, rating: str, comment: str | None) -> str:
    settings = get_settings()
    feedback_id = str(uuid.uuid4())
    record = {
        "id": feedback_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    with settings.feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return feedback_id
