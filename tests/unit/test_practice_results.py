from uuid import uuid4

import pytest

from app.services.practice_results import (
    PracticeResultValidationError,
    calculate_attempt_summary,
    calculate_wrong_count_delta,
    group_item_results,
    group_is_correct,
)


@pytest.mark.parametrize(
    ("answers", "mastered"),
    [
        ([True], True),
        ([False], False),
        ([True, True, True], True),
        ([True, False, True], False),
    ],
)
def test_group_mastery_requires_every_item_correct(answers, mastered):
    assert group_is_correct(answers) is mastered


@pytest.mark.parametrize(
    ("previous", "next_value", "expected"),
    [
        (None, False, 1),
        (None, True, 0),
        (False, False, 0),
        (False, True, -1),
        (True, False, 1),
        (True, True, 0),
    ],
)
def test_wrong_count_delta_only_applies_state_difference(
    previous, next_value, expected
):
    assert calculate_wrong_count_delta(previous, next_value) == expected


def test_grouping_requires_exactly_one_answer_for_every_sheet_item():
    question_id = uuid4()
    item_id = uuid4()
    items = [{"id": item_id, "wrong_question_id": question_id}]

    with pytest.raises(PracticeResultValidationError, match="missing"):
        group_item_results(items, {})

    with pytest.raises(PracticeResultValidationError, match="extra"):
        group_item_results(items, {item_id: True, uuid4(): False})


def test_summary_counts_items_and_groups_by_source_question():
    first_question = uuid4()
    second_question = uuid4()
    item_ids = [uuid4() for _ in range(3)]
    items = [
        {"id": item_ids[0], "wrong_question_id": first_question},
        {"id": item_ids[1], "wrong_question_id": first_question},
        {"id": item_ids[2], "wrong_question_id": second_question},
    ]
    groups = group_item_results(
        items,
        {item_ids[0]: True, item_ids[1]: False, item_ids[2]: True},
    )

    summary = calculate_attempt_summary(groups, correct_count=2, total_count=3)

    assert summary.correct_count == 2
    assert summary.total_count == 3
    assert str(summary.accuracy) == "0.6667"
    assert [group.all_correct for group in summary.groups] == [False, True]
