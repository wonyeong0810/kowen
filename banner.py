#!/usr/bin/env python3
"""유튜브 채널 배너 (2560x1440). 모든 기기에서 보이는 안전 영역(가운데 1546x423)에 로고와 문구를 넣는다."""
import math
import random

from PIL import Image, ImageDraw

import kowen_news as k

W, H, S = 320, 180, 8          # 2560 x 1440
SAFE = (507, 508, 2053, 931)   # 유튜브 텍스트·로고 안전 영역


def globe(low, cx, cy, R):
    g = Image.new("RGB", (45, 45), (40, 110, 210))
    gd = ImageDraw.Draw(g)
    green = (60, 170, 80)
    gd.polygon([(6, 6), (18, 4), (20, 10), (14, 16), (12, 20), (16, 24), (20, 28), (18, 36), (14, 42), (10, 34),
                (8, 24), (4, 16)], fill=green)
    gd.polygon([(26, 6), (36, 4), (40, 10), (34, 14), (30, 12)], fill=green)
    gd.polygon([(28, 16), (38, 16), (42, 22), (38, 32), (34, 38), (30, 30), (26, 22)], fill=green)
    gd.ellipse([20, 1, 26, 4], fill=(240, 240, 250))
    g = g.resize((2 * R + 1, 2 * R + 1), Image.NEAREST)
    m = Image.new("L", g.size, 0)
    ImageDraw.Draw(m).ellipse([0, 0, 2 * R, 2 * R], fill=255)
    low.paste(g, (cx - R, cy - R), m)
    ImageDraw.Draw(low).ellipse([cx - R, cy - R, cx + R, cy + R], outline=(230, 230, 255), width=2)


def main():
    low = Image.new("RGB", (W, H), (12, 16, 40))
    d = ImageDraw.Draw(low)
    for i in range(32):                                       # 방사선 배경
        a = i / 32 * 6.283
        c = (20, 30, 70) if i % 2 else (14, 20, 50)
        d.polygon([(160, 90), (160 + math.cos(a) * 400, 90 + math.sin(a) * 400),
                   (160 + math.cos(a + 0.1) * 400, 90 + math.sin(a + 0.1) * 400)], fill=c)
    rnd = random.Random(4)                                    # 아래쪽 도시 야경 (TV 화면에서만 보임)
    for bx in range(0, W, 14):
        bh = rnd.randint(14, 34)
        k.rect(d, bx, H - bh, 12, bh, (22, 26, 55))
        for wy in range(H - bh + 3, H - 2, 5):
            for wx in range(bx + 2, bx + 11, 4):
                if rnd.random() < 0.45:
                    k.rect(d, wx, wy, 2, 2, (255, 220, 120))
    globe(low, 96, 90, 22)                                    # 안전 영역 왼쪽
    img = low.resize((W * S, H * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.text((1000, 570), "KOWEN", font=k.font(240), fill=(255, 210, 60), stroke_width=14, stroke_fill=(0, 0, 0))
    d.rectangle([1000, 820, 1460, 916], fill=(200, 30, 40), outline=(0, 0, 0), width=8)
    d.text((1230, 826), "NEWS", font=k.font(96), fill=(255, 255, 255), anchor="ma")
    d.text((1500, 846), "Today's top stories", font=k.font(48), fill=(220, 225, 255))
    img.save("out/banner.png")
    prev = img.copy()                                         # 안전 영역 표시 미리보기
    ImageDraw.Draw(prev).rectangle(SAFE, outline=(255, 0, 255), width=6)
    prev.resize((1280, 720)).save("out/banner_safe_preview.png")
    print("완료: out/banner.png")


if __name__ == "__main__":
    main()
