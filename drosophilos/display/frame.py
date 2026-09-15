"""The display: pixel records (x, y, colour) from the substrate into a PNG. The host's role is
to show what the machine computed; the palette is the only thing it adds."""

from __future__ import annotations

from PIL import Image

PALETTE = {0: (0, 0, 0), 1: (72, 72, 96), 2: (96, 72, 40), 3: (220, 100, 60), 4: (190, 84, 48), 5: (150, 66, 36),
           6: (110, 50, 26), 7: (72, 34, 18), 8: (255, 255, 255), 9: (200, 200, 0), 10: (0, 160, 0), 11: (0, 120, 200),
           12: (160, 0, 160), 13: (0, 180, 180), 14: (128, 128, 128), 15: (255, 0, 0)}


def write_png(pixels: dict, width: int, height: int, path: str, scale: int = 4) -> None:
    """`pixels`: (x, y) -> colour index; missing pixels are black."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = PALETTE.get(pixels.get((x, y), 0) & 15, (0, 0, 0))
    if scale > 1:
        img = img.resize((width * scale, height * scale), Image.NEAREST)
    img.save(path)
