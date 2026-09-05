import numpy as np
import pytest
from tupper import codec

def random_pixels(width, height, depth, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << depth, size=(height, width), dtype=np.uint8)

@pytest.mark.parametrize("width, height, depth", [
    (1, 1, 1),
    (8, 8, 2),
    (64, 64, 2),
    (128, 128, 4),
    (17, 43, 6),
    (32, 32, 8),
])
def test_round_trip(width, height, depth):
    pixels = random_pixels(width, height, depth)
    recovered, got_depth = codec.decode(codec.encode(pixels, depth))
    
    assert got_depth == depth
    assert np.array_equal(recovered, pixels)
    
def test_header_survives_leading_zeros():
    pixels = random_pixels(1, 4, 1)
    recovered, _ = codec.decode(codec.encode(pixels, 1))
    
    assert recovered.shape == (4, 1)
    assert np.array_equal(recovered, pixels)
    
def test_all_zero_image_still_decodes():
    pixels = np.zeros((16, 16), dtype=np.uint8)
    k = codec.encode(pixels, 4)
    recovered, _ = codec.decode(k)
    
    assert k > 0
    assert not recovered.any()
    
def test_stream_length_matches_header():
    pixels =  random_pixels(20, 12, 4)
    k = codec.encode(pixels, 4)
    assert k.bit_length() == codec.PREFIX_BITS + 20 * 12 * 4
    
def test_non_square_keeps_orientation():
    pixels = random_pixels(31, 19, 6)
    recovered, _ = codec.decode(codec.encode(pixels, 6))
    assert recovered.shape == (19, 31)
    
def test_rejects_values_above_depth():
    pixels = np.array([[0, 4]], dtype=np.uint8)
    with pytest.raises(codec.CodecError):
        codec.encode(pixels, 2)
        
def test_rejects_oversized_dimensions():
    pixels = np.zeros((1, codec.MAX_DIM + 1), dtype=np.uint8)
    with pytest.raises(codec.CodecError):
        codec.encode(pixels, 1)
        
@pytest.mark.parametrize("k", [0, -5])
def test_rejects_non_positive(k):
    with pytest.raises(codec.CodecError):
        codec.decode(k)
        
def test_rejects_truncated_stream():
    pixels = random_pixels(8, 8, 4)
    k = codec.encode(pixels, 4)
    with pytest.raises(codec.CodecError):
        codec.decode(k >> 8)
        
def test_rejects_unknown_version():
    pixels = random_pixels(8, 8, 4)
    k = codec.encode(pixels, 4)
    bumped = k ^ (1 << (k.bit_length() - 2))
    with pytest.raises(codec.CodecError):
        codec.decode(bumped)