def segment_questions(image_path: str, ocr_lines: list) -> list[dict]:
    """
    基于 OCR 行间距和编号（1. 2. ① ② 等）切分题目区域。
    返回 cropped regions: [{"index": 0, "bbox": [x,y,w,h], "text_lines": [...]}, ...]
    """
    # 策略：按大间距（>40px）或题号标记切分
    regions = []
    current_region = []
    prev_bottom = 0

    for i, line in enumerate(ocr_lines):
        bbox = line["bbox"]
        top = min(p[1] for p in bbox)
        bottom = max(p[1] for p in bbox)

        if current_region and (top - prev_bottom > 40):  # 大间距=新题
            regions.append(_build_region(current_region))
            current_region = []
        current_region.append(line)
        prev_bottom = bottom

    if current_region:
        regions.append(_build_region(current_region))
    return regions

def _build_region(lines: list) -> dict:
    tops = [min(p[1] for p in l["bbox"]) for l in lines]
    bottoms = [max(p[1] for p in l["bbox"]) for l in lines]
    lefts = [min(p[0] for p in l["bbox"]) for l in lines]
    rights = [max(p[0] for p in l["bbox"]) for l in lines]
    return {
        "bbox": [min(lefts), min(tops), max(rights)-min(lefts), max(bottoms)-min(tops)],
        "text_lines": lines,
        "text": "\n".join(l["text"] for l in lines),
    }
