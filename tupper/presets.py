from dataclasses import dataclass

@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    size: int
    depth: int
    
    @property
    def levels(self):
        return 1 << self.depth
    
    @property
    def stream_bits(self):
        return self.size * self.size * self.depth
    
PRESETS = {p.key: p for p in [
    Preset("sketch", "Sketch", 64, 2),
    Preset("standard", "Standard", 128, 4),
    Preset("fine", "Fine", 192, 6),
    Preset("sharp", "Sharp", 320, 6),
    Preset("maximum", "Maximum", 512, 8),
]}

DEFAULT = "standard"

ORDER = list(PRESETS)

def get(key):
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError("unknown preset %r" % key) from None
