"""Deterministic retry merging and boundary metadata for vision diagnostics."""

from __future__ import annotations


def build_unit_ocr_tokens(unit: dict, ocr_lines: list[dict]) -> list[str]:
    """Return punctuation-insensitive character n-grams for a unit's OCR text."""
    line_ids = {int(value) for value in unit.get("ocr_line_ids", [])}
    normalized = "".join(
        character.casefold()
        for line in sorted(
            ocr_lines,
            key=lambda item: int(item.get("ocr_line_id", 0)),
        )
        if int(line.get("ocr_line_id", -1)) in line_ids
        for character in str(line.get("text", ""))
        if character.isalnum()
    )
    if len(normalized) < 2:
        return [normalized] if normalized else []
    return sorted({normalized[index : index + 2] for index in range(len(normalized) - 1)})


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _bbox_iou(first: list[float], second: list[float]) -> float:
    intersection = _intersection_area(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union else 0.0


def _bbox_containment(first: list[float], second: list[float]) -> float:
    intersection = _intersection_area(first, second)
    smaller = min(_bbox_area(first), _bbox_area(second))
    return intersection / smaller if smaller else 0.0


def _set_jaccard(first, second) -> float:
    first_set = set(first)
    second_set = set(second)
    union = first_set | second_set
    return len(first_set & second_set) / len(union) if union else 0.0


def _new_cluster(event: dict) -> dict:
    unit_id = str(event["question_unit_id"])
    source_round = int(event["source_round"])
    return {
        "question_unit_id": unit_id,
        "unit_bbox": list(event["unit_bbox"]),
        "anchor_ids": sorted({int(value) for value in event.get("anchor_ids", [])}),
        "ocr_tokens": sorted({str(value) for value in event.get("ocr_tokens", [])}),
        "boundary_fits": sorted(
            {str(value) for value in event.get("boundary_fits", [])}
        ),
        "source_rounds": [source_round],
        "source_unit_ids": [unit_id],
        "source_events": [{"round": source_round, "question_unit_id": unit_id}],
    }


def _merge_into(cluster: dict, event: dict) -> None:
    event_unit_id = str(event["question_unit_id"])
    event_bbox = list(event["unit_bbox"])
    cluster["anchor_ids"] = sorted(
        set(cluster["anchor_ids"]) | {int(value) for value in event.get("anchor_ids", [])}
    )
    cluster["ocr_tokens"] = sorted(
        set(cluster["ocr_tokens"]) | {str(value) for value in event.get("ocr_tokens", [])}
    )
    cluster["boundary_fits"] = sorted(
        set(cluster["boundary_fits"])
        | {str(value) for value in event.get("boundary_fits", [])}
    )
    cluster["source_rounds"] = sorted(
        set(cluster["source_rounds"]) | {int(event["source_round"])}
    )
    cluster["source_unit_ids"] = sorted(
        set(cluster["source_unit_ids"]) | {event_unit_id}
    )
    cluster["source_events"].append(
        {"round": int(event["source_round"]), "question_unit_id": event_unit_id}
    )
    if (_bbox_area(event_bbox), event_unit_id) < (
        _bbox_area(cluster["unit_bbox"]),
        cluster["question_unit_id"],
    ):
        cluster["question_unit_id"] = event_unit_id
        cluster["unit_bbox"] = event_bbox


def _merge_clusters(target: dict, source: dict) -> None:
    target["anchor_ids"] = sorted(set(target["anchor_ids"]) | set(source["anchor_ids"]))
    target["ocr_tokens"] = sorted(set(target["ocr_tokens"]) | set(source["ocr_tokens"]))
    target["boundary_fits"] = sorted(
        set(target["boundary_fits"]) | set(source["boundary_fits"])
    )
    target["source_rounds"] = sorted(
        set(target["source_rounds"]) | set(source["source_rounds"])
    )
    target["source_unit_ids"] = sorted(
        set(target["source_unit_ids"]) | set(source["source_unit_ids"])
    )
    target["source_events"].extend(source["source_events"])
    if (_bbox_area(source["unit_bbox"]), source["question_unit_id"]) < (
        _bbox_area(target["unit_bbox"]),
        target["question_unit_id"],
    ):
        target["question_unit_id"] = source["question_unit_id"]
        target["unit_bbox"] = list(source["unit_bbox"])


def consolidate_retry_events(rounds: list[list[dict]], config: dict) -> dict:
    """Union retry results, then conservatively merge duplicate question units."""
    min_iou = float(config["retry_duplicate_min_iou"])
    min_containment = float(config["retry_duplicate_min_containment"])
    min_ocr = float(config["retry_duplicate_min_ocr_jaccard"])
    stable_by_id = {}
    exact_merges = 0
    decisions = []
    for round_events in rounds:
        for event in round_events:
            unit_id = str(event["question_unit_id"])
            exact = stable_by_id.get(unit_id)
            if exact is not None:
                _merge_into(exact, event)
                exact_merges += 1
                decisions.append(
                    {"question_unit_id": unit_id, "action": "merge_exact_id"}
                )
                continue
            stable_by_id[unit_id] = _new_cluster(event)
    stable_union = sorted(
        stable_by_id.values(),
        key=lambda item: (
            item["unit_bbox"][1],
            item["unit_bbox"][0],
            item["question_unit_id"],
        ),
    )
    clusters = []
    conservative_merges = 0
    for event in stable_union:
        unit_id = str(event["question_unit_id"])
        duplicate = None
        metrics = None
        for cluster in clusters:
            iou = _bbox_iou(cluster["unit_bbox"], event["unit_bbox"])
            containment = _bbox_containment(
                cluster["unit_bbox"], event["unit_bbox"]
            )
            ocr_jaccard = _set_jaccard(
                cluster["ocr_tokens"], event.get("ocr_tokens", [])
            )
            shared_anchor = bool(
                set(cluster["anchor_ids"]) & set(event.get("anchor_ids", []))
            )
            if (
                iou >= min_iou
                and ocr_jaccard >= min_ocr
                and (shared_anchor or containment >= min_containment)
            ):
                duplicate = cluster
                metrics = {
                    "iou": round(iou, 6),
                    "containment": round(containment, 6),
                    "ocr_jaccard": round(ocr_jaccard, 6),
                    "shared_anchor": shared_anchor,
                }
                break
        if duplicate is None:
            clusters.append({**event, "source_events": list(event["source_events"])})
            decisions.append(
                {"question_unit_id": unit_id, "action": "keep_distinct"}
            )
            continue
        representative_before = duplicate["question_unit_id"]
        _merge_clusters(duplicate, event)
        conservative_merges += 1
        decisions.append(
            {
                "question_unit_id": unit_id,
                "action": "merge_ocr_geometry",
                "representative_before": representative_before,
                "metrics": metrics,
            }
        )
    clusters.sort(
        key=lambda item: (
            item["unit_bbox"][1],
            item["unit_bbox"][0],
            item["question_unit_id"],
        )
    )
    return {
        "stable_id_union_events": stable_union,
        "events": clusters,
        "audit": {
            "input_event_count": sum(len(items) for items in rounds),
            "output_event_count": len(clusters),
            "exact_id_merge_count": exact_merges,
            "conservative_ocr_geometry_merge_count": conservative_merges,
            "decisions": decisions,
        },
    }


def build_boundary_metadata(
    events: list[dict],
    units_by_id: dict[str, dict],
    anchor_candidates: dict[str, list[dict]],
    config: dict,
) -> list[dict]:
    """Choose a safe narrow display unit and expose only ambiguous local options."""
    min_iou = float(config["boundary_option_min_iou"])
    min_containment = float(config["boundary_option_min_containment"])
    min_ocr = float(config["boundary_auto_narrow_min_ocr_jaccard"])
    maximum_options = int(config["boundary_interaction_max_options"])
    results = []
    for event in events:
        selected_id = str(event["question_unit_id"])
        selected = units_by_id[selected_id]
        selected_bbox = list(selected["unit_bbox"])
        selected_ocr = selected.get("ocr_tokens", event.get("ocr_tokens", []))
        ranked_ids = []
        all_candidate_ids = []
        rank_one_containing_ids = []
        selected_excludes_anchor = False
        non_primary_anchor_unit = False
        straddled_anchor_window = False
        selected_center_y = (selected_bbox[1] + selected_bbox[3]) / 2
        for anchor_id in event.get("anchor_ids", []):
            anchor_options = anchor_candidates.get(str(anchor_id), [])
            for candidate in anchor_options:
                candidate_id = str(candidate["question_unit_id"])
                if candidate_id not in all_candidate_ids:
                    all_candidate_ids.append(candidate_id)
            selected_option = next(
                (
                    candidate
                    for candidate in anchor_options
                    if str(candidate["question_unit_id"]) == selected_id
                ),
                None,
            )
            if selected_option is not None and not selected_option.get("anchor_in_unit"):
                selected_excludes_anchor = True
            rank_one = next(
                (candidate for candidate in anchor_options if candidate.get("anchor_in_unit")),
                None,
            )
            if rank_one is not None:
                rank_one_id = str(rank_one["question_unit_id"])
                if rank_one_id not in rank_one_containing_ids:
                    rank_one_containing_ids.append(rank_one_id)
                if (
                    selected_option is not None
                    and selected_option.get("anchor_in_unit")
                    and rank_one_id != selected_id
                ):
                    non_primary_anchor_unit = True
            containing_centers = []
            for candidate in anchor_options:
                candidate_id = str(candidate["question_unit_id"])
                candidate_unit = units_by_id.get(candidate_id)
                if not candidate.get("anchor_in_unit") or candidate_unit is None:
                    continue
                candidate_bbox = candidate_unit["unit_bbox"]
                if _bbox_iou(selected_bbox, candidate_bbox) < min_iou:
                    continue
                containing_centers.append(
                    (candidate_bbox[1] + candidate_bbox[3]) / 2
                )
            if (
                any(value < selected_center_y for value in containing_centers)
                and any(value > selected_center_y for value in containing_centers)
            ):
                straddled_anchor_window = True
            for candidate in anchor_options:
                candidate_id = str(candidate["question_unit_id"])
                if candidate.get("anchor_in_unit") and candidate_id not in ranked_ids:
                    ranked_ids.append(candidate_id)
        if selected_id not in ranked_ids:
            ranked_ids.insert(0, selected_id)
        local_options = []
        matching_narrow = []
        conflicting = []
        for candidate_id in ranked_ids:
            candidate = units_by_id.get(candidate_id)
            if candidate is None:
                continue
            bbox = list(candidate["unit_bbox"])
            if candidate_id != selected_id:
                if _bbox_iou(selected_bbox, bbox) < min_iou:
                    continue
                if _bbox_containment(selected_bbox, bbox) < min_containment:
                    continue
            local_options.append(candidate_id)
            ocr_jaccard = _set_jaccard(
                selected_ocr, candidate.get("ocr_tokens", [])
            )
            if ocr_jaccard >= min_ocr:
                matching_narrow.append(candidate_id)
            elif candidate_id != selected_id:
                conflicting.append(candidate_id)
        display_id = min(
            matching_narrow or [selected_id],
            key=lambda unit_id: (
                _bbox_area(units_by_id[unit_id]["unit_bbox"]),
                unit_id,
            ),
        )
        ordered_options = [display_id] + [
            unit_id for unit_id in local_options if unit_id != display_id
        ]
        multi_anchor_conflict = len(rank_one_containing_ids) > 1
        if multi_anchor_conflict:
            ordered_options.extend(
                unit_id
                for unit_id in rank_one_containing_ids
                if unit_id not in ordered_options and unit_id in units_by_id
            )
        if non_primary_anchor_unit:
            ordered_options.extend(
                unit_id
                for unit_id in rank_one_containing_ids
                if unit_id not in ordered_options and unit_id in units_by_id
            )
        if straddled_anchor_window:
            ordered_options.extend(
                unit_id
                for unit_id in ranked_ids
                if unit_id not in ordered_options and unit_id in units_by_id
            )
        if selected_excludes_anchor:
            ordered_options.extend(
                unit_id
                for unit_id in all_candidate_ids
                if unit_id not in ordered_options and unit_id in units_by_id
            )
        model_risk = "sibling_intrusion" in set(event.get("boundary_fits", []))
        reasons = (
            (["different_ocr_anchor_containing_option"] if conflicting else [])
            + (["model_boundary_risk"] if model_risk else [])
            + (["multi_anchor_rank_one_conflict"] if multi_anchor_conflict else [])
            + (["non_primary_anchor_unit"] if non_primary_anchor_unit else [])
            + (["straddled_anchor_window"] if straddled_anchor_window else [])
            + (["selected_unit_excludes_anchor"] if selected_excludes_anchor else [])
        )
        results.append(
            {
                **event,
                "display_unit_id": display_id,
                "display_bbox": list(units_by_id[display_id]["unit_bbox"]),
                "boundary_status": (
                    "boundary_ambiguous" if reasons else "automatic"
                ),
                "boundary_reasons": sorted(reasons),
                "interaction_option_unit_ids": ordered_options[:maximum_options],
            }
        )
    return results
