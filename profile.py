#!/usr/bin/env python3
"""유튜브 채널 프로필 사진 (800x800, 원형으로 잘려도 안전하게)."""
from PIL import Image, ImageDraw

import kowen_news as k

N, S = 100, 8


def main():
    low = Image.new("RGB", (N, N), (12, 16, 40))
    d = ImageDraw.Draw(low)
    for i in range(12):                                   # 은은한 방사선 배경
        import math
        a = i / 12 * 6.283
        c = (20, 30, 70) if i % 2 else (14, 20, 50)
        d.polygon([(50, 40), (50 + math.cos(a) * 90, 40 + math.sin(a) * 90),
                   (50 + math.cos(a + 0.26) * 90, 40 + math.sin(a + 0.26) * 90)], fill=c)
    cx, cy, R = 50, 38, 22                                 # 픽셀 지구본 (대서양 쪽에서 본 모습)
    globe = Image.new("RGB", (2 * R + 1, 2 * R + 1), (40, 110, 210))
    g = ImageDraw.Draw(globe)
    green = (60, 170, 80)
    g.polygon([(6, 6), (18, 4), (20, 10), (14, 16), (12, 20), (16, 24), (20, 28), (18, 36), (14, 42), (10, 34),
               (8, 24), (4, 16)], fill=green)                     # 아메리카
    g.polygon([(26, 6), (36, 4), (40, 10), (34, 14), (30, 12)], fill=green)   # 유럽
    g.polygon([(28, 16), (38, 16), (42, 22), (38, 32), (34, 38), (30, 30), (26, 22)], fill=green)   # 아프리카
    g.ellipse([20, 1, 26, 4], fill=(240, 240, 250))               # 그린란드/극지방
    mask = Image.new("L", globe.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 2 * R, 2 * R], fill=255)
    low.paste(globe, (cx - R, cy - R), mask)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(230, 230, 255), width=2)
    img = low.resize((N * S, N * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.text((400, 520), "KOWEN", font=k.font(112), fill=(255, 210, 60), anchor="ma",
           stroke_width=6, stroke_fill=(0, 0, 0))
    d.rectangle([290, 640, 510, 696], fill=(200, 30, 40))
    d.text((400, 644), "NEWS", font=k.font(56), fill=(255, 255, 255), anchor="ma")
    img.save("out/profile.png")
    # 원형 미리보기
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 799, 799], fill=255)
    prev = Image.new("RGB", img.size, (255, 255, 255))
    prev.paste(img, (0, 0), mask)
    prev.save("out/profile_circle_preview.png")
    print("완료: out/profile.png")


if __name__ == "__main__":
    main()
