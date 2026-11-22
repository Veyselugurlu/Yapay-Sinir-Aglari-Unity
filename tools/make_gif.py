"""
Frames recorded by XorVisualizer -> README animation and a film strip for the report.
  Visualizer.exe -seed 10 -recordFrames <dir> -recordCount 280 -screen-width 960 -screen-height 540 -screen-fullscreen 0
  python tools/make_gif.py <dir>
The visualizer trains 4 epochs per frame, so frame i shows the network after 4 (i + 1) epochs.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EPOCHS_PER_FRAME = 4
STRIP_EPOCHS = [4, 244, 444, 524, 644, 1000]

frames = [Image.open(p).convert("RGB") for p in sorted(Path(sys.argv[1]).glob("frame_*.png"))]
out = Path("docs/week09/images")
out.mkdir(parents=True, exist_ok=True)

last = 1000 // EPOCHS_PER_FRAME  # training stops at epoch 1000; hold the final picture a little
small = [f.resize((480, 270), Image.LANCZOS).quantize(64) for f in frames[:last:2] + [frames[last - 1]] * 15]
small[0].save(out / "xor_training.gif", save_all=True, append_images=small[1:], duration=80, loop=0, optimize=True)

font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
w, h = 400, 225
strip = Image.new("RGB", (3 * w + 40, 2 * (h + 28) + 10), (255, 255, 255))
draw = ImageDraw.Draw(strip)
for k, epoch in enumerate(STRIP_EPOCHS):
    i = min(epoch // EPOCHS_PER_FRAME - 1, len(frames) - 1)
    x, y = 10 + (k % 3) * (w + 10), 5 + (k // 3) * (h + 28)
    strip.paste(frames[i].resize((w, h), Image.LANCZOS), (x, y))
    draw.text((x + 4, y + h + 5), f"{epoch}. epoch", fill=(30, 30, 30), font=font)
strip.save(out / "xor_filmstrip.png")
print((out / "xor_training.gif").stat().st_size // 1024, "KB")
