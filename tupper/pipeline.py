"""Between image files and the codec"""
import io
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from . import codec
from .dither import quantize, expand

class ImageError(Exception):
    pass

def load(source):
    """Input image -> Output image Upright & Grayscale."""
    try:
        image = Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("could not read that file as an image") from exc
    
    image = ImageOps.exif_transpose(image)
    return image.convert("L")

def center_square(image):
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))

def prepare(image, preset):
    """Fix image to preset dimensions, including cropping, dithering and grayscaling."""
    square = center_square(image).resize((preset.size, preset.size), Image.LANCZOS)
    return quantize(np.asarray(square), preset.levels)

def encode_image(image, preset):
    return codec.encode(prepare(image, preset), preset.depth)

def to_png(pixels, depth, scale=1):
    """Render pixel indices as a PNG"""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    
    image = Image.fromarray(expand(pixels, 1 << depth), mode="L")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
        
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()

def render(k, scale=1):
    pixels, depth = codec.decode(k)
    return to_png(pixels, depth, scale)