"""Download and validate the configured RapidOCR models during image build."""

import argparse
from pathlib import Path

from rapidocr import (
    EngineType,
    LangDet,
    LangRec,
    ModelType,
    OCRVersion,
    RapidOCR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    RapidOCR(
        params={
            "Global.model_root_dir": str(model_path),
            "Global.log_level": "warning",
            "Det.engine_type": EngineType(args.engine),
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType(args.model_type),
            "Det.ocr_version": OCRVersion(args.model_version),
            "Rec.engine_type": EngineType(args.engine),
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": ModelType(args.model_type),
            "Rec.ocr_version": OCRVersion(args.model_version),
        }
    )
    if not list(model_path.rglob("*.onnx")):
        raise RuntimeError("RapidOCR ONNX models were not prepared")


if __name__ == "__main__":
    main()
