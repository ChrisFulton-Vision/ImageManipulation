from PIL import Image
from dataclasses import dataclass
from cv2 import cvtColor, COLOR_BGR2RGB
from support.core.enums import ExportQuality


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


def make_gif(images, fps=10, name='output', infinite: bool = False,
             quality: ExportQuality = ExportQuality.med_quality):
    # dirList = sorted(os.listdir('ImagesToGif'), key=numerical_sort)
    pil_images = []

    match quality:
        case ExportQuality.hgh_quality:
            compress = CompressionSettings.high_res()
        case ExportQuality.med_quality:
            compress = CompressionSettings.med_res()
        case ExportQuality.low_quality:
            compress = CompressionSettings.low_res()
        case _:
            compress = CompressionSettings.low_res()

    for idx, cv_img in enumerate(images):
        pil_img = Image.fromarray(cvtColor(cv_img, COLOR_BGR2RGB))
        pil_img = pil_img.convert(mode=compress.mode,
                                  palette=compress.palette,
                                  colors=compress.colors,
                                  dither=compress.dither)
        pil_images.append(pil_img)

    dur = max(1, round(1000 / fps))
    pil_images[0].save(
        name + '.gif',
        save_all=True,
        append_images=pil_images[1:],
        duration=dur,  # Duration in milliseconds between frames
        loop=0 if infinite else 1,  # 0 for infinite loop
        optimize=compress.optimize
    )

def make_apng(images, fps=10, name='output', infinite: bool = False,
              quality: ExportQuality = ExportQuality.med_quality):
    """
    Drop-in sibling to make_gif() with the *same* call signature / quality selector,
    but writes an Animated PNG (APNG).

    Output file will be: f"{name}.png"
    """
    pil_images = []

    match quality:
        case ExportQuality.hgh_quality:
            print("high res")
            compress = CompressionSettings.high_res()
        case ExportQuality.med_quality:
            print("med res")
            compress = CompressionSettings.med_res()
        case ExportQuality.low_quality:
            print("low res")
            compress = CompressionSettings.low_res()
        case _:
            compress = CompressionSettings.low_res()

    # Build frames
    for cv_img in images:
        pil_img = Image.fromarray(cvtColor(cv_img, COLOR_BGR2RGB))

        # APNG does NOT need palette quantization like GIF.
        # But if your CompressionSettings includes a target mode (e.g., "RGB"/"RGBA"),
        # respect it without forcing palette conversion.
        target_mode = getattr(compress, "mode", None)
        if target_mode and pil_img.mode != target_mode:
            # Only convert if it's a sane PNG mode
            if target_mode in ("RGB", "RGBA", "L"):
                pil_img = pil_img.convert(mode=target_mode)

        pil_images.append(pil_img)

    if not pil_images:
        raise ValueError("make_apng: no images provided")

    # True ms timing (no GIF 10ms tick nonsense)
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"make_apng: fps must be > 0, got {fps}")
    dur_ms = max(1, round(1000.0 / fps))

    # APNG is saved as a PNG file with save_all=True
    out_path = name + '.png'
    pil_images[0].save(
        out_path,
        format="PNG",
        save_all=True,
        append_images=pil_images[1:],
        duration=dur_ms,               # milliseconds per frame
        loop=0 if infinite else 1,     # 0 = infinite
        optimize=getattr(compress, "optimize", False),
    )

    return out_path