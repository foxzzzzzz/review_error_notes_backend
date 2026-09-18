"""Pure evidence contracts for Chinese marked-evidence recognition output."""

from copy import deepcopy
from math import isclose, isfinite
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator
from sqlalchemy import or_, select

from app.models.wrong_question import WrongQuestion


PIPELINE_NAME = "chinese_marked_evidence_v1"


EvidenceSourceClass = Literal[
    "printed", "student_handwriting", "teacher_correction", "unknown"
]
EvidenceRole = Literal[
    "printed_instruction",
    "printed_prompt",
    "printed_pinyin",
    "printed_hanzi",
    "student_answer",
    "teacher_correction",
    "unknown",
]
EvidenceStatus = Literal["confirmed", "supported", "conflict", "insufficient"]
AnswerStatus = Literal["suggested", "unresolved"]
MarkStatus = Literal["confirmed", "needs_review"]


class _StrictEvidenceModel(BaseModel):
    """Base contract for provider-independent evidence data."""

    model_config = ConfigDict(strict=True, extra="forbid")


class EvidenceObservation(_StrictEvidenceModel):
    """One text observation from a named recognition provider."""

    source: Literal["deepseek", "ocr"]
    text: StrictStr
    bbox: list[StrictFloat] = Field(min_length=4, max_length=4)
    source_class: EvidenceSourceClass
    confidence: StrictFloat = Field(ge=0, le=1)
    role: EvidenceRole = "unknown"

    @model_validator(mode="after")
    def _require_ordered_bbox(self):
        x1, y1, x2, y2 = self.bbox
        if not all(isfinite(value) for value in self.bbox):
            raise ValueError("bbox coordinates must be finite")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must have positive area in x1, y1, x2, y2 order")
        expected_source_classes = {
            "printed_instruction": "printed",
            "printed_prompt": "printed",
            "printed_pinyin": "printed",
            "printed_hanzi": "printed",
            "student_answer": "student_handwriting",
            "teacher_correction": "teacher_correction",
        }
        expected_source_class = expected_source_classes.get(self.role)
        if (
            expected_source_class is not None
            and self.source_class != expected_source_class
        ):
            raise ValueError(
                f"role {self.role} requires source_class {expected_source_class}"
            )
        if self.source == "ocr" and self.role != "unknown":
            raise ValueError("OCR observations must retain an unknown explicit role")
        return self


class FieldEvidence(_StrictEvidenceModel):
    """All observations and a deterministic status for one spatial field."""

    field: Literal[
        "printed_instruction",
        "printed_prompt",
        "printed_pinyin",
        "printed_hanzi",
        "student_answer",
        "teacher_correction",
        "unknown",
    ]
    status: EvidenceStatus
    selected_value: StrictStr | None
    observations: list[EvidenceObservation]
    conflicts: list[StrictStr]


class AnswerSuggestion(_StrictEvidenceModel):
    """Automatic answers are suggestions only until a person reviews them."""

    status: AnswerStatus
    selected_value: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_selected_value(self):
        if self.status == "suggested" and not self.selected_value:
            raise ValueError("suggested answers require a selected value")
        if self.status == "unresolved" and self.selected_value is not None:
            raise ValueError("unresolved answers cannot select a value")
        return self


class EvidenceIdentity(_StrictEvidenceModel):
    """Identity and unmodified geometry for a single marked question."""

    image_id: StrictStr | StrictInt
    mark_id: StrictInt
    question_geometry: dict[str, Any]


class RoleConflict(_StrictEvidenceModel):
    """One explicit provider role contradicted by independent answer geometry."""

    claimed_role: Literal[
        "printed_instruction",
        "printed_prompt",
        "printed_pinyin",
        "printed_hanzi",
    ]
    conflicting_role: Literal["student_answer"]
    reason: Literal["printed_observation_overlaps_student_answer_bbox"]
    observation: EvidenceObservation


class EvidenceBundle(_StrictEvidenceModel):
    """Serializable, review-only evidence assembled without network or storage I/O."""

    schema_version: Literal[1] = 1
    identity: EvidenceIdentity
    mark_status: MarkStatus
    question_evidence_status: EvidenceStatus
    fields: dict[StrictStr, FieldEvidence]
    answer_suggestion: AnswerSuggestion
    role_conflicts: list[RoleConflict] = Field(default_factory=list)


_ROLE_BBOX_KEYS = {
    "student_answer": "student_answer_bbox",
    "teacher_correction": "teacher_correction_bbox",
    "printed_instruction": "printed_instruction_bbox",
    "printed_prompt": "printed_prompt_bbox",
    "printed_pinyin": "printed_pinyin_bbox",
    "printed_hanzi": "printed_hanzi_bbox",
}

_PRINTED_FIELDS = (
    "printed_instruction",
    "printed_prompt",
    "printed_pinyin",
    "printed_hanzi",
)


def _as_observation(value: EvidenceObservation | dict[str, Any]) -> EvidenceObservation:
    if isinstance(value, EvidenceObservation):
        return value
    return EvidenceObservation.model_validate(value)


def _ordered_bbox(value: Any) -> list[float] | None:
    """Return a usable bbox only when structure supplied an ordered rectangle."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not isfinite(item)
        for item in value
    ):
        return None
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def _overlaps(left: list[float], right: list[float]) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(
        left[1], right[1]
    ) < min(left[3], right[3])


def _validated_structure_observation(
    structure_observation: dict[str, Any],
) -> dict[str, Any]:
    for bbox_key in _ROLE_BBOX_KEYS.values():
        if bbox_key in structure_observation and _ordered_bbox(
            structure_observation[bbox_key]
        ) is None:
            raise ValueError(f"{bbox_key} must be a finite bbox with positive area")
    return structure_observation


def _validated_role_routing_hints(
    role_routing_hints: dict[str, Any] | None,
) -> dict[str, list[float]]:
    if role_routing_hints is None:
        return {}
    if not isinstance(role_routing_hints, dict):
        raise TypeError("role_routing_hints must be a dict or None")
    unsupported_roles = sorted(
        set(role_routing_hints) - set(_PRINTED_FIELDS),
        key=repr,
    )
    if unsupported_roles:
        raise ValueError(
            "role_routing_hints contains unsupported roles: "
            + ", ".join(repr(role) for role in unsupported_roles)
        )
    validated = {}
    for role in sorted(role_routing_hints):
        bbox = _ordered_bbox(role_routing_hints[role])
        if bbox is None:
            raise ValueError(
                f"role_routing_hints[{role}] must be a finite bbox with positive area"
            )
        validated[role] = bbox
    return validated


def _smallest_unambiguous_overlapping_role(
    observation_bbox: list[float],
    role_bboxes: dict[str, list[float]],
) -> str | None:
    candidates = sorted(
        (
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            role,
        )
        for role, bbox in role_bboxes.items()
        if _overlaps(observation_bbox, bbox)
    )
    if not candidates:
        return None
    smallest_area = candidates[0][0]
    smallest_roles = [
        role
        for area, role in candidates
        if isclose(area, smallest_area, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return smallest_roles[0] if len(smallest_roles) == 1 else None


def _role_for_observation(
    observation: EvidenceObservation,
    structure_observation: dict[str, Any],
    role_routing_hints: dict[str, list[float]],
) -> str:
    """Assign an evidence field without turning routing hints into confirmation."""
    if observation.role != "unknown":
        return observation.role
    if observation.source_class == "teacher_correction":
        return "teacher_correction"
    if observation.source_class == "student_handwriting":
        return "student_answer"
    observation_bbox = [float(value) for value in observation.bbox]
    student_answer_bbox = _ordered_bbox(
        structure_observation.get("student_answer_bbox")
    )
    if student_answer_bbox is not None and _overlaps(
        observation_bbox, student_answer_bbox
    ):
        return "student_answer"
    if observation.source == "ocr":
        routed_role = _smallest_unambiguous_overlapping_role(
            observation_bbox,
            role_routing_hints,
        )
        if routed_role is not None:
            return routed_role
    structure_role_bboxes = {}
    for field, bbox_key in _ROLE_BBOX_KEYS.items():
        if field == "student_answer":
            continue
        role_bbox = _ordered_bbox(structure_observation.get(bbox_key))
        if role_bbox is not None:
            structure_role_bboxes[field] = role_bbox
    structured_role = _smallest_unambiguous_overlapping_role(
        observation_bbox,
        structure_role_bboxes,
    )
    if structured_role is not None:
        return structured_role
    return {
        "printed": "printed_prompt",
        "student_handwriting": "student_answer",
        "teacher_correction": "teacher_correction",
        "unknown": "unknown",
    }[observation.source_class]


def _spatially_compatible(
    field: str,
    observations: list[EvidenceObservation],
    structure_observation: dict[str, Any],
) -> bool:
    bbox_key = _ROLE_BBOX_KEYS.get(field)
    role_bbox = (
        _ordered_bbox(structure_observation.get(bbox_key)) if bbox_key else None
    )
    return role_bbox is not None and all(
        _overlaps([float(value) for value in observation.bbox], role_bbox)
        for observation in observations
    )


def _field_evidence(
    field: str,
    observations: list[EvidenceObservation],
    structure_observation: dict[str, Any],
) -> FieldEvidence:
    usable = [observation for observation in observations if observation.text.strip()]
    values = sorted({observation.text.strip() for observation in usable})
    if not values:
        status: EvidenceStatus = "insufficient"
        selected_value = None
        conflicts: list[str] = []
    elif len(values) > 1:
        status = "conflict"
        selected_value = None
        conflicts = values
    else:
        sources = {observation.source for observation in usable}
        status = (
            "confirmed"
            if sources == {"deepseek", "ocr"}
            and _spatially_compatible(field, usable, structure_observation)
            else "supported"
        )
        selected_value = values[0]
        conflicts = []
    return FieldEvidence(
        field=field,
        status=status,
        selected_value=selected_value,
        observations=observations,
        conflicts=conflicts,
    )


def assemble_question_evidence(
    *,
    image_id: str | int,
    mark_id: int,
    question_geometry: dict[str, Any],
    deepseek_observations: Sequence[EvidenceObservation | dict[str, Any]],
    ocr_observations: Sequence[EvidenceObservation | dict[str, Any]],
    structure_observation: dict[str, Any],
    role_routing_hints: dict[str, Any] | None = None,
    suggested_answer_candidate: str | None = None,
    mark_status: str | None = None,
) -> EvidenceBundle:
    """Assemble evidence by source class and declared spatial role, never by text guesses."""
    structure_observation = _validated_structure_observation(structure_observation)
    role_routing_hints = _validated_role_routing_hints(role_routing_hints)
    if suggested_answer_candidate is not None and not isinstance(
        suggested_answer_candidate, str
    ):
        raise TypeError("suggested_answer_candidate must be a string or None")
    grouped: dict[str, list[EvidenceObservation]] = {
        **{field: [] for field in _PRINTED_FIELDS},
        "student_answer": [],
        "teacher_correction": [],
        "unknown": [],
    }
    role_conflicts = []
    for expected_source, raw_observations in (
        ("deepseek", deepseek_observations),
        ("ocr", ocr_observations),
    ):
        for raw_observation in raw_observations:
            observation = _as_observation(raw_observation)
            if observation.source != expected_source:
                raise ValueError(
                    f"{expected_source}_observations must contain source={expected_source}"
                )
            student_answer_bbox = _ordered_bbox(
                structure_observation.get("student_answer_bbox")
            )
            explicit_printed_role_conflict = (
                observation.role in _PRINTED_FIELDS
                and student_answer_bbox is not None
                and _overlaps(
                    [float(value) for value in observation.bbox],
                    student_answer_bbox,
                )
            )
            if explicit_printed_role_conflict:
                field = "unknown"
                role_conflicts.append(
                    {
                        "claimed_role": observation.role,
                        "conflicting_role": "student_answer",
                        "reason": "printed_observation_overlaps_student_answer_bbox",
                        "observation": observation.model_dump(mode="json"),
                    }
                )
            else:
                field = _role_for_observation(
                    observation,
                    structure_observation,
                    role_routing_hints,
                )
            if field == "student_answer" and observation.source != "ocr":
                observation = observation.model_copy(
                    update={"source_class": "student_handwriting"}
                )
            grouped.setdefault(field, []).append(observation)

    fields = {
        field: _field_evidence(field, observations, structure_observation)
        for field, observations in grouped.items()
    }
    populated_printed_fields = [
        fields[field]
        for field in _PRINTED_FIELDS
        if fields[field].status != "insufficient"
    ]
    if role_conflicts or any(
        field.status == "conflict" for field in populated_printed_fields
    ):
        question_evidence_status: EvidenceStatus = "conflict"
    elif not populated_printed_fields:
        question_evidence_status = "insufficient"
    elif all(field.status == "confirmed" for field in populated_printed_fields):
        question_evidence_status = "confirmed"
    else:
        question_evidence_status = "supported"
    candidate = (suggested_answer_candidate or "").strip()
    answer_suggestion = (
        AnswerSuggestion(status="suggested", selected_value=candidate)
        if candidate
        else AnswerSuggestion(status="unresolved")
    )
    return EvidenceBundle(
        identity=EvidenceIdentity(
            image_id=image_id,
            mark_id=mark_id,
            question_geometry=deepcopy(question_geometry),
        ),
        mark_status="confirmed" if mark_status == "confirmed" else "needs_review",
        question_evidence_status=question_evidence_status,
        fields=fields,
        answer_suggestion=answer_suggestion,
        role_conflicts=role_conflicts,
    )


def build_pending_evidence_values(
    bundle: EvidenceBundle,
    student_text: str | None,
    question_type: str | None,
    crop_region: dict[str, Any] | None,
    subject: str | None = None,
    tags: Sequence[str] | None = None,
    difficulty: int | None = None,
    raw_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build fixed pending-review database values from a pure evidence bundle."""
    answer_status = bundle.answer_suggestion.status
    collection_status = "pending_review"
    review_status = "needs_review"
    assert answer_status != "confirmed"
    assert collection_status == "pending_review"
    raw_json = deepcopy(raw_evidence) if raw_evidence else {}
    raw_json["evidence_bundle"] = bundle.model_dump(mode="json")
    return {
        "recognition_pipeline": PIPELINE_NAME,
        "mark_status": bundle.mark_status,
        "question_evidence_status": bundle.question_evidence_status,
        "answer_status": answer_status,
        "collection_status": collection_status,
        "review_status": review_status,
        "ocr_text": student_text or None,
        "ocr_answer": bundle.answer_suggestion.selected_value,
        "subject": subject,
        "question_type": question_type or None,
        "tags": list(tags or []),
        "difficulty": difficulty,
        "crop_region": deepcopy(crop_region),
        "ocr_raw_json": raw_json,
    }


def _positive_bbox_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return (
        max(left[0], right[0]) < min(left[2], right[2])
        and max(left[1], right[1]) < min(left[3], right[3])
    )


def audit_primary_page_cv_coverage(
    candidates: Sequence[dict[str, Any]],
    regions: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Record local red-region coverage without changing primary candidates."""
    valid_regions = [
        bbox
        for bbox in (_ordered_bbox(region) for region in regions)
        if bbox is not None
    ]
    covered_region_indexes = set()
    for candidate in candidates:
        crop_region = candidate.get("crop_region") or {}
        model_bbox = _ordered_bbox(crop_region.get("model_bbox"))
        covered_indexes = (
            [
                index
                for index, region in enumerate(valid_regions)
                if _positive_bbox_overlap(model_bbox, region)
            ]
            if model_bbox is not None
            else []
        )
        covered_region_indexes.update(covered_indexes)
        raw_json = candidate.setdefault("ocr_raw_json", {})
        raw_json["local_cv_audit"] = {
            "candidate_bbox": model_bbox,
            "covered_region_indexes": covered_indexes,
        }
    uncovered_region_indexes = [
        index for index in range(len(valid_regions)) if index not in covered_region_indexes
    ]
    diagnostic = {
        "candidate_count": len(candidates),
        "region_count": len(valid_regions),
        "covered_region_indexes": sorted(covered_region_indexes),
        "uncovered_region_indexes": uncovered_region_indexes,
        "requires_manual_review": bool(uncovered_region_indexes),
    }
    if uncovered_region_indexes:
        for candidate in candidates:
            raw_json = candidate.setdefault("ocr_raw_json", {})
            raw_json["local_cv_audit"]["page_uncovered_region_indexes"] = (
                uncovered_region_indexes
            )
            reasons = raw_json.setdefault("page_review_reasons", [])
            if "local_cv_uncovered_strong_red_mark" not in reasons:
                reasons.append("local_cv_uncovered_strong_red_mark")
    return diagnostic


def is_downstream_eligible(question) -> bool:
    """Keep legacy records usable while requiring confirmation for this pipeline."""
    return (
        getattr(question, "recognition_pipeline", None) != PIPELINE_NAME
        or getattr(question, "answer_status", None) == "confirmed"
    )


def downstream_eligibility_clause():
    """Return the SQL equivalent of :func:`is_downstream_eligible`."""
    return or_(
        WrongQuestion.recognition_pipeline.is_(None),
        WrongQuestion.recognition_pipeline != PIPELINE_NAME,
        WrongQuestion.answer_status == "confirmed",
    )


def eligible_questions_statement(question_ids, student_id):
    """Select collected, owned questions that may enter downstream flows."""
    return select(WrongQuestion).where(
        WrongQuestion.id.in_(question_ids),
        WrongQuestion.student_id == student_id,
        WrongQuestion.deleted_at.is_(None),
        WrongQuestion.collection_status == "collected",
        downstream_eligibility_clause(),
    )
