"""
transforming images into a single integer, and reading them.
bit stream: 1(sentinel) | version[4bits] | width[12bits] | height[12bits] | depth[4bits] | img data 
pixel data is column-major.
"""

import numpy as np

VERSION = 1

HEADER_BITS = 32
PREFIX_BITS = HEADER_BITS + 1

MAX_DIM = (1 << 12) - 1
MAX_DEPTH = (1 << 4) - 1

class CodecError(Exception):
    pass

def _pack_header(width, height, depth):
    if not 1 <= width <= MAX_DIM or not 1 <= height <= MAX_DIM:
        raise CodecError("dimensions must fit in 12 bits")
    if not 1 <= depth <= MAX_DEPTH:
        raise CodecError("depth must fit in 4 bits")
    
    value = VERSION << 28 | width << 16 | height << 4 | depth
    return np.array([(value >> i) & 1 for i in range(HEADER_BITS - 1, -1, -1)],
                    dtype=np.uint8)
    
def _unpack_header(bits):
    value = 0
    for bit in bits:
        value = value << 1 | int(bit)
        
    version = value >> 28
    if version != VERSION:
        raise CodecError("unsupported stream version %d" % version)
    
    return (value >> 16) & 0xFFF, (value >> 4) & 0xFFF, value & 0xF

def encode(pixels, depth):
    """Turn quantized pixel indices into the constant that draws them."""
    if pixels.ndim != 2:
        raise CodecError("expected a 2-D array of pixel indices")
    if pixels.max(initial=0) >= 1 << depth:
        raise CodecError("pixel values exceed the declared depth")
    
    height, width = pixels.shape
    planes = np.unpackbits((pixels << (8 - depth))[:, :, None], axis=2, count=depth, bitorder="big")
    body = planes.transpose(1, 0, 2).reshape(-1)
    
    stream = np.concatenate([[1], _pack_header(width, height, depth), body])
    
    padding = -len(stream) % 8
    packed = np.packbits(np.concatenate([stream, np.zeros(padding, np.uint8)]))
    return int.from_bytes(packed.tobytes(), "big") >> padding

def decode (k):
    """Recover pixel indices and depth from a constant"""
    if not isinstance(k, int) or k <= 0:
        raise CodecError("k must be a positive integer")
    
    length = k.bit_length()
    if length <= PREFIX_BITS:
        raise CodecError("stream is too short to hold a header")
    
    raw = np.frombuffer(k.to_bytes((length + 7) // 8, "big"), np.uint8)
    bits = np.unpackbits(raw)[-length:]
    
    width, height, depth = _unpack_header(bits[1:PREFIX_BITS])
    
    body = bits[PREFIX_BITS:]
    wanted = width * height * depth
    if len(body) != wanted:
        raise CodecError("stream holds %d pixel bits, header wants %d" % (len(body), wanted))
    
    planes = body.reshape(width, height, depth).transpose(1, 0, 2)
    weights = (1 << np.arange(depth - 1, -1, -1)).astype(np.uint16)
    return (planes * weights).sum(axis=2).astype(np.uint8), depth