import numpy as np

def bayer(order):
    """ٌRecursive Bayer threshold matrix, 2**order on a side."""
    m = np.zeros((1, 1))
    for _ in range(order):
        m = np.block([[4 * m, 4 * m + 2],
                      [4 * m + 3, 4 * m + 1]])
    return m

_THRESHOLD = (bayer(3) + 0.5) / 64.0
_TILE = _THRESHOLD.shape[0]

def quantize(grey, levels):
    """Reduce an 8-bit grayscale array to `levels` shades,
    ordered dither. Returns indices in 0..levels-1, not 
    brightness values."""
    if levels < 2:
        raise ValueError("need at least two levels")
    
    height, width = grey.shape
    reps = (height // _TILE + 1, width // _TILE + 1)
    tile = np.tile(_THRESHOLD, reps)[:height, :width]
    
    scaled = grey.astype(np.float64) / 255.0 * (levels - 1)
    return np.clip(np.floor(scaled + tile), 0, levels - 1).astype(np.uint8)