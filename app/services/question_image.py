import math
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


class QuestionImageError(Exception):
    pass


class QuestionImageNotFound(QuestionImageError):
    pass


class QuestionImageInvalid(QuestionImageError):
    pass


def _pixel_crop_box(image_size, crop_region):
    if not isinstance(crop_region, dict):
        return None
    if crop_region.get("bbox_format") != "normalized_ltrb":
        return None

    bbox = crop_region.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return None
    if not all(math.isfinite(value) for value in bbox):
        return None

    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return None

    width, height = image_size
    pixel_left = max(0, min(width - 1, math.floor(left * width)))
    pixel_top = max(0, min(height - 1, math.floor(top * height)))
    pixel_right = max(pixel_left + 1, min(width, math.ceil(right * width)))
    pixel_bottom = max(pixel_top + 1, min(height, math.ceil(bottom * height)))
    return pixel_left, pixel_top, pixel_right, pixel_bottom


def render_question_image(image_path, crop_region, view, jpeg_quality, max_pixels):
    path = Path(image_path)
    if not path.is_file():
        raise QuestionImageNotFound("Question image does not exist")

    try:
        with Image.open(path) as source:
            if source.width * source.height > max_pixels:
                raise QuestionImageInvalid("Question image exceeds the pixel limit")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if view == "crop":
                crop_box = _pixel_crop_box(image.size, crop_region)
                if crop_box:
                    image = image.crop(crop_box)

            output = BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality)
            return output.getvalue()
    except Image.DecompressionBombError as exc:
        raise QuestionImageInvalid("Question image exceeds the pixel limit") from exc
    except (OSError, ValueError) as exc:
        raise QuestionImageInvalid("Question image is invalid") from exc
