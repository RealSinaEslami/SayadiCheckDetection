import os
import random
import uuid 
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, features

# ==================== تنظیمات ====================
BASE_IMAGE_PATH = "in_payment_of.png"
FONTS_DIR = Path("fonts")

OUTPUT_DIR = Path("generated_checks_huge")
ZIP_NAME = "persian_filled_checks_huge.zip"

TOTAL_COUNT = 500
BOX_X = 420
BOX_Y = 130        
BOX_WIDTH = 400    
BOX_HEIGHT = 150   

SCALE_FACTOR = 6   
PADDING_X = 2 * SCALE_FACTOR
PADDING_Y = 2 * SCALE_FACTOR

HAS_RAQM = features.check("raqm")
if not HAS_RAQM:
    import arabic_reshaper
    from bidi.algorithm import get_display

def prepare_persian_text(text: str) -> str:
    if HAS_RAQM: return text
    return get_display(arabic_reshaper.reshape(text))

NAMES = [
    "شرکت تهران گستر فراز", "شرکت بهمن گستر ایران", "تهران پلیمر صبا", "آرمان توسعه",
    "آرمان گستر ایران", "شرکت توسعه گستر", "هوشمند کبیر", "آرمان توسعه هوش", "آرمان هوشمند کبیر",
    "توسعه هوشمند", "توسعه کبیر"
    "علی رضایی", "زهرا احمدی", "محمد حسینی", "مریم موسوی", "امیرحسین کریمی",
    "سارا محمدی", "حسین مرادی", "نگار جعفری", "پارسا رحمانی", "نرگس قاسمی",
]

def get_available_fonts(folder: Path):
    font_files = list(folder.glob("*.ttf")) + list(folder.glob("*.otf")) + list(folder.glob("*.TTF")) + list(folder.glob("*.OTF"))
    return [str(f) for f in font_files]

AVAILABLE_FONTS = get_available_fonts(FONTS_DIR)

def get_text_dimensions(draw, text: str, font):
    if HAS_RAQM:
        bbox = draw.textbbox((0, 0), text, font=font, direction="rtl", language="fa")
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def get_fitted_font(draw, text: str, font_path: str, max_w: int, max_h: int):
    size = int(90 * SCALE_FACTOR) 
    min_size = int(30 * SCALE_FACTOR)
    while size >= min_size:
        try:
            font = ImageFont.truetype(font_path, size=size)
        except Exception:
            break
        w, h = get_text_dimensions(draw, text, font)
        if w <= max_w and h <= max_h:
            return font, (w, h)
        size -= 2
    font = ImageFont.truetype(font_path, size=min_size)
    w, h = get_text_dimensions(draw, text, font)
    return font, (w, h)

def render_name_on_base_image(base_img: Image.Image, name: str, font_path: str) -> Path:
    canvas = base_img.copy().convert("RGBA")
    local_w, local_h = BOX_WIDTH * SCALE_FACTOR, BOX_HEIGHT * SCALE_FACTOR
    txt_canvas = Image.new("RGBA", (local_w, local_h), (255, 255, 255, 0))
    txt_draw = ImageDraw.Draw(txt_canvas)

    processed_text = prepare_persian_text(name)
    max_w, max_h = local_w - (2 * PADDING_X), local_h - (2 * PADDING_Y)
    font, (tw, th) = get_fitted_font(txt_draw, processed_text, font_path, max_w, max_h)

    TEXT_Y_OFFSET = -70 * SCALE_FACTOR

    x = (local_w - tw) // 2
    y = (local_h - th) // 2 + TEXT_Y_OFFSET

    y = max(PADDING_Y, y)
    y = min(y, local_h - th - PADDING_Y)

    ink = random.randint(5, 20)
    text_color = (ink, ink, ink, 255)

    if HAS_RAQM:
        txt_draw.text((x, y), processed_text, font=font, fill=text_color, direction="rtl", language="fa")
    else:
        txt_draw.text((x, y), processed_text, font=font, fill=text_color)

    angle = random.uniform(-0.8, 0.8)
    rotated = txt_canvas.rotate(angle, resample=Image.BICUBIC, expand=False)
    final_text_layer = rotated.resize((BOX_WIDTH, BOX_HEIGHT), resample=Image.LANCZOS)

    canvas.paste(final_text_layer, (BOX_X, BOX_Y), mask=final_text_layer)
    out = canvas.convert("RGB")
    
    # فقط UUID
    unique_filename = f"{uuid.uuid4()}.png"
    out_path = OUTPUT_DIR / unique_filename
    out.save(out_path, format="PNG", optimize=True)
    return out_path

def main():
    if not os.path.exists(BASE_IMAGE_PATH): 
        raise FileNotFoundError(f"تصویر {BASE_IMAGE_PATH} پیدا نشد")
    
    base_image = Image.open(BASE_IMAGE_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    last_font = None

    # حلقه ۵۰۰ تایی با انتخاب تصادفی نام
    for idx in range(1, TOTAL_COUNT + 1):
        name = random.choice(NAMES)
        
        choices = [f for f in AVAILABLE_FONTS if f != last_font] if len(AVAILABLE_FONTS) > 1 else AVAILABLE_FONTS
        font_path = random.choice(choices)
        last_font = font_path
        
        out_path = render_name_on_base_image(base_image, name, font_path)
        generated.append(out_path)
        print(f"[{idx:03d}/{TOTAL_COUNT}] {name} -> {out_path.name}")

    with zipfile.ZipFile(ZIP_NAME, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in generated: 
            zf.write(p, arcname=p.name)

    print(f"تمام شد! ۵۰۰ تصویر تولید و در {ZIP_NAME} ذخیره شدند.")

if __name__ == "__main__":
    main()
