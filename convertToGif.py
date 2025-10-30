from PIL import Image, ImageDraw
from dataclasses import dataclass
import os, cv2


@dataclass
class CompressionSettings:
    # If mode is None, we skip convert() entirely.
    mode: str | None
    palette: int | None
    colors: int | None
    dither: int
    optimize: bool

    @classmethod
    def high_res(cls):
        # High quality for GIF usually = adaptive 256 + FS dither + optimize
        return cls(
            mode='P',
            palette=Image.ADAPTIVE,
            colors=256,
            dither=Image.FLOYDSTEINBERG,
            optimize=True
        )

    @classmethod
    def med_res(cls):
        return cls(
            mode='P',
            palette=Image.ADAPTIVE,
            colors=128,
            dither=Image.FLOYDSTEINBERG,
            optimize=True
        )

    @classmethod
    def low_res(cls):
        # Smaller file: fewer colors, no dither, still optimize
        return cls(
            mode='P',
            palette=Image.ADAPTIVE,
            colors=64,
            dither=Image.NONE,
            optimize=True
        )


def numerical_sort(file_name):
    try:
        return int(file_name.split('.')[0])
    except (ValueError, IndexError):
        return float('inf')

def make_gif(images, fps = 10, name='output', infinite: bool = False, quality='h'):
    # dirList = sorted(os.listdir('ImagesToGif'), key=numerical_sort)
    pil_images = []

    match(quality):
        case 'h':
            compress = CompressionSettings.high_res()
        case 'm':
            compress = CompressionSettings.med_res()
        case 'l':
            compress = CompressionSettings.low_res()
        case _:
            compress = CompressionSettings.low_res()

    for idx, cv_img in enumerate(images):
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.convert(mode=compress.mode,
                                  palette=compress.palette,
                                  colors=compress.colors,
                                  dither=compress.dither)
        pil_images.append(pil_img)

    dur = int(1000 / fps)
    pil_images[0].save(
        name + '.gif',
        save_all=True,
        append_images=pil_images[1:],
        duration=dur,  # Duration in milliseconds between frames
        loop= 0 if infinite else 1,  # 0 for infinite loop
        optimize=compress.optimize
    )