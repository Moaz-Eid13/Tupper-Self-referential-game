import io
import numpy as np
import pytest
from PIL import Image
from tupper import codec, presets
from tupper.pipeline import (ImageError, center_square, encode_image,
                            load, prepare, render, to_png)

def gradient(width, height):
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (height, 1)), mode="L")

def as_file(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer

def test_preset_square_in_codec_limits():
    for preset in presets.PRESETS.values():
        assert preset.size <= codec.MAX_DIM
        assert preset.depth <= codec.MAX_DEPTH
        assert preset.levels == 2 ** preset.depth
        
def test_default_preset():
    assert presets.DEFAULT in presets.PRESETS
    
def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        presets.get("test")
        
def test_square_center_centered():
    strip = Image.fromarray(np.arange(100, dtype=np.uint8).reshape(1, 100),
            mode="L").resize((100, 20), Image.NEAREST)
    cropped = center_square(strip)
    
    assert cropped.size == (20, 20)
    assert np.asarray(cropped)[0, 0] == 40
    
def test_center_square_keep_size():
    square = gradient(30, 30)
    assert center_square(square).size == (30, 30)
    
@pytest.mark.parametrize("key", list(presets.PRESETS))
def test_prepare_matches_preset(key):
    preset = presets.get(key)
    pixels = prepare(gradient(400, 300), preset)
    
    assert pixels.shape == (preset.size, preset.size)
    assert pixels.max() < preset.levels
    
def test_encode_round_through_codec():
    preset = presets.get("sketch")
    image = gradient(200, 200)
    
    k = encode_image(image, preset)
    recovered, depth = codec.decode(k)
    
    assert depth == preset.depth
    assert np.array_equal(recovered, prepare(image, preset))
    
def test_load_to_grayscale():
    color = Image.new("RGB", (10, 10), (255, 0, 0))
    assert load(as_file(color)).mode == "L"
    
def test_load_junk():
    with pytest.raises(ImageError):
        load(io.BytesIO(b"this is not an image"))
        
def test_to_png_size():
    pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    data = to_png(pixels, 4, scale=3)
    
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(data)).size == (12, 12)
    
def test_to_png_range():
    pixels = np.array([[0, 3]], dtype=np.uint8)
    shades = np.asarray(Image.open(io.BytesIO(to_png(pixels, 2))))
    
    assert shades[0, 0] == 0
    assert shades[0, 1] == 255
    
def test_to_png_shrinking():
    with pytest.raises(ValueError):
        to_png(np.zeros((4, 4), dtype=np.uint8), 4, scale=0)
        
def test_render():
    preset = presets.get("sketch")
    image = gradient(150, 90)
    
    k = encode_image(image, preset)
    shown = np.asarray(Image.open(io.BytesIO(render(k))))
    
    assert shown.shape == (preset.size, preset.size)
    assert np.array_equal(shown, np.asarray(
        Image.open(io.BytesIO(to_png(prepare(image, preset), preset.depth)))
    ))