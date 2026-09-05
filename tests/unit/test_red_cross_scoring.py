import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from scripts.diagnose_vision_pipeline import detect_red_cross_candidates


CONFIG = json.loads((Path(__file__).parents[2] / 'scripts/cv_cross_experiment_config.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('kind', ['blank', 'random', 'edges', 'dense'])
def test_acceleration_preserves_candidates_and_all_masks(tmp_path, kind):
    rng = np.random.default_rng(731)
    pixels = np.full((111, 143, 3), 230, dtype=np.uint8)
    if kind == 'random':
        pixels[rng.random(pixels.shape[:2]) < 0.35] = [200, 30, 30]
    elif kind == 'dense':
        pixels[:] = [190, 50, 40]
    image = Image.fromarray(pixels)
    if kind == 'edges':
        draw = ImageDraw.Draw(image)
        for x, y in [(0, 0), (142, 110), (72, 55), (3, 60)]:
            draw.line((x-9, y-9, x+9, y+9), fill=(220, 30, 30), width=3)
            draw.line((x-9, y+9, x+9, y-9), fill=(220, 30, 30), width=3)
    path = tmp_path / 'page.png'
    image.save(path)
    reference = detect_red_cross_candidates(path, CONFIG)
    accelerated = detect_red_cross_candidates(path, CONFIG, vectorized=True)
    for key in ['candidates', 'red_pixel_count', 'analysis_width', 'analysis_height']:
        assert accelerated[key] == reference[key]
    for key in ['red_mask', 'geometry_mask', 'candidate_center_mask']:
        assert np.array_equal(accelerated[key], reference[key]), key
