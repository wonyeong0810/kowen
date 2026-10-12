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
    k.draw_person(d, 50, 70, 1.0, False, k.ANCHOR, gesture=False)   # 앵커
    k.rect(d, 18, 66, 64, 4, (255, 200, 60))                          # 책상
    k.rect(d, 22, 70, 56, 14, (35, 45, 90))
    img = low.resize((N * S, N * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.text((400, 568), "KOWEN", font=k.font(112), fill=(255, 210, 60), anchor="ma",
           stroke_width=6, stroke_fill=(0, 0, 0))
    d.rectangle([300, 680, 500, 728], fill=(200, 30, 40))
    d.text((400, 682), "NEWS", font=k.font(48), fill=(255, 255, 255), anchor="ma")
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
