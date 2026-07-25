from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from uuid import UUID


class PracticeResultValidationError(ValueError):
    pass


@dataclass(frozen=True)
class QuestionGroupResult:
    wrong_question_id: UUID | None
    item_ids: tuple[UUID, ...]
    all_correct: bool


@dataclass(frozen=True)
class AttemptSummary:
    correct_count: int
    total_count: int
    accuracy: Decimal
    groups: tuple[QuestionGroupResult, ...]


def group_is_correct(answers) -> bool:
    values = tuple(answers)
    return bool(values) and all(values)


def calculate_wrong_count_delta(
    previous_group_correct: bool | None,
    next_group_correct: bool,
) -> int:
    if previous_group_correct is None:
        return 0 if next_group_correct else 1
    if previous_group_correct == next_group_correct:
        return 0
    return -1 if next_group_correct else 1


def _value(item, key):
    return item[key] if isinstance(item, dict) else getattr(item, key)


def _source_question_id(item):
    question_id = _value(item, "wrong_question_id")
    if question_id is not None:
        return question_id
    snapshot = _value(item, "question_snapshot") or {}
    value = snapshot.get("source_wrong_question_id")
    return UUID(value) if value else None


def group_item_results(items, answers) -> tuple[QuestionGroupResult, ...]:
    item_rows = tuple(items)
    if not item_rows:
        raise PracticeResultValidationError("empty sheet")

    expected_ids = [_value(item, "id") for item in item_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise PracticeResultValidationError("duplicate sheet item IDs")

    answer_ids = set(answers)
    missing = set(expected_ids) - answer_ids
    extra = answer_ids - set(expected_ids)
    if missing:
        raise PracticeResultValidationError("missing sheet item answers")
    if extra:
        raise PracticeResultValidationError("extra sheet item answers")

    grouped = {}
    for item in item_rows:
        question_id = _source_question_id(item)
        grouped.setdefault(question_id, []).append(_value(item, "id"))

    return tuple(
        QuestionGroupResult(
            wrong_question_id=question_id,
            item_ids=tuple(item_ids),
            all_correct=group_is_correct(answers[item_id] for item_id in item_ids),
        )
        for question_id, item_ids in grouped.items()
    )


def calculate_attempt_summary(
    groups,
    *,
    correct_count: int,
    total_count: int,
) -> AttemptSummary:
    if total_count <= 0:
        raise PracticeResultValidationError("empty sheet")
    if correct_count < 0 or correct_count > total_count:
        raise PracticeResultValidationError("invalid correct count")
    accuracy = (Decimal(correct_count) / Decimal(total_count)).quantize(
        Decimal("0.0001"),
        rounding=ROUND_HALF_UP,
    )
    return AttemptSummary(
        correct_count=correct_count,
        total_count=total_count,
        accuracy=accuracy,
        groups=tuple(groups),
    )


def build_review_groups(items, result_by_item):
    grouped = {}
    for item in items:
        question_id = _source_question_id(item)
        snapshot = _value(item, "question_snapshot") or {}
        grouped.setdefault(question_id, []).append(
            {
                "sheet_item_id": _value(item, "id"),
                "question_type": snapshot.get(
                    "question_type",
                    _value(item, "question_type"),
                ),
                "question_text": snapshot.get(
                    "question_text",
                    _value(item, "question_text"),
                ),
                "sort_order": snapshot.get(
                    "sort_order",
                    _value(item, "sort_order"),
                ),
                "is_correct": result_by_item.get(_value(item, "id"), True),
            }
        )
    return [
        {"wrong_question_id": question_id, "items": group_items}
        for question_id, group_items in grouped.items()
    ]


def apply_group_result(
    question,
    *,
    all_correct: bool,
    previous_correct: bool | None,
    now,
    update_mastery: bool = True,
):
    question.wrong_count = max(
        1,
        (question.wrong_count or 1)
        + calculate_wrong_count_delta(previous_correct, all_correct),
    )
    if not update_mastery:
        return

    question.last_practiced_at = now
    if all_correct:
        question.mastery_status = "mastered"
        question.mastered_at = now
    else:
        question.mastery_status = "learning"
        question.mastered_at = None
