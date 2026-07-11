from paddleocr import PaddleOCR
import json

ocr = PaddleOCR(lang="ch", use_angle_cls=True)

def recognize_text(image_path: str) -> dict:
    """识别整张图片，返回文本和区域信息"""
    results = ocr.ocr(image_path, cls=True)
    lines = []
    full_text = ""
    if results and results[0]:
        for line in results[0]:
            bbox, (text, confidence) = line
            lines.append({"bbox": bbox, "text": text, "confidence": confidence})
            full_text += text + "\n"
    return {"lines": lines, "full_text": full_text.strip()}
