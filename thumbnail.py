#!/usr/bin/env python3
"""KOWEN News 영상 표지(썸네일) 생성기. 1280x720 PNG.

사용법:
    python3 thumbnail.py            # out/*_thumb.png 전부 생성
"""
import os

from PIL import Image, ImageDraw

import kowen_news as k

TW, TH = 160, 90   # 표지는 8배 확대해서 픽셀을 더 굵게
S = 8


def big_text(d, xy, txt, size, fill, stroke=8, anchor="la"):
    d.text(xy, txt, font=k.font(size), fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0), anchor=anchor)


def badge(d):
    d.rectangle([40, 36, 380, 112], fill=(255, 200, 60))
    d.rectangle([40, 112, 380, 120], fill=(200, 30, 40))
    d.text((58, 42), "KOWEN NEWS", font=k.font(64), fill=(15, 20, 50))


def finish(low, draw_text, path):
    img = low.resize((TW * S, TH * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    draw_text(d)
    badge(d)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print("완료:", path)


def ocean():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, TW, TH, (30, 90, 170))
    for y in range(4, TH, 8):
        for x in range(0, TW, 12):
            k.rect(d, x + (y % 12), y, 4, 1, (70, 140, 215))
    d.polygon([(130, 0), (160, 0), (160, 90), (140, 90), (128, 60), (136, 30)], fill=(70, 160, 80))
    k.rect(d, 112, 44, 1, 26, (230, 230, 230))      # 깃발
    k.rect(d, 113, 44, 16, 9, (255, 205, 60))
    k.rect(d, 115, 47, 12, 2, (200, 30, 40))
    k.draw_trump(d, 40, 92, 0.05)                   # 손가락 하늘 찌르는 포즈

    def text(d):
        big_text(d, (1240, 300), "ATLANTIC", 96, (255, 255, 255), anchor="ra")
        x0, y0, x1, y1 = d.textbbox((1240, 300), "ATLANTIC", font=k.font(96), anchor="ra")
        d.rectangle([x0 - 10, (y0 + y1) // 2 - 6, x1 + 10, (y0 + y1) // 2 + 6], fill=(230, 30, 40))
        big_text(d, (1240, 420), "TRUMP", 176, (255, 210, 60), stroke=10, anchor="ra")
        big_text(d, (1240, 580), "OCEAN?!", 144, (255, 210, 60), stroke=10, anchor="ra")

    finish(low, text, "out/trump_ocean_thumb.png")


def diesel():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.draw_flag(d, "ru", 80, 0, 80, 90)
    k.rect(d, 0, 0, 80, 90, (40, 50, 130))
    for i in range(0, 90, 10):
        k.rect(d, 0, i, 80, 5, (190, 30, 40))
    k.rect(d, 78, 0, 4, 90, (10, 10, 15))
    k.draw_trump(d, 40, 92, 0.05)
    k.draw_person(d, 120, 92, 0, False, k.PUTIN, gesture=False)
    # 가운데 디젤 주유기
    k.rect(d, 70, 64, 20, 26, (230, 230, 235))
    k.rect(d, 73, 67, 14, 6, (20, 20, 30))
    k.rect(d, 72, 76, 16, 6, (250, 200, 40))
    k.rect(d, 90, 70, 4, 3, (30, 30, 30))

    def text(d):
        big_text(d, (640, 150), "RUSSIA SANCTIONS", 112, (255, 255, 255), anchor="ma")
        big_text(d, (640, 270), "EASED", 176, (255, 210, 60), stroke=10, anchor="ma")
        d.rectangle([470, 440, 810, 500], fill=(200, 30, 40))
        d.text((640, 446), "DIESEL DEAL", font=k.font(48), fill=(255, 255, 255), anchor="ma")

    finish(low, text, "out/russia_diesel_thumb.png")


def dmz():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.vgrad(d, 0, 50, (60, 20, 20), (150, 60, 30))
    d.polygon([(0, 50), (40, 30), (90, 48), (130, 28), (160, 44), (160, 50)], fill=(50, 40, 35))
    k.rect(d, 0, 50, TW, 40, (80, 95, 55))
    for r, c in ((26, (255, 120, 40)), (18, (255, 190, 80)), (9, (255, 245, 200))):   # 폭발
        d.ellipse([80 - r, 52 - r // 1.4, 80 + r, 52 + r // 1.4], fill=c)
    k.rect(d, 0, 58, TW, 1, (160, 160, 160))                  # 철책
    for fx in range(0, TW, 10):
        k.rect(d, fx, 48, 1, 16, (110, 110, 110))
    for fx in range(0, TW, 3):
        k.rect(d, fx, 52 + (fx // 3) % 2, 2, 1, (180, 180, 180))
    d.polygon([(64, 80), (84, 80), (74, 64)], fill=(220, 30, 30))   # 지뢰 경고판
    k.rect(d, 73, 70, 2, 5, (255, 255, 255))
    k.rect(d, 73, 76, 2, 2, (255, 255, 255))
    k.draw_char(d, 30, 92, 0.05, False, "soldier")
    k.draw_char(d, 132, 92, 0.05, False, "kim_jong_un")

    def text(d):
        f = k.font(144)
        w1, w2 = d.textlength("KOREAN WAR 2", font=f), d.textlength("?!", font=f)
        x = (1280 - w1 - w2) / 2
        big_text(d, (x, 150), "KOREAN WAR 2", 144, (255, 255, 255), stroke=10)
        big_text(d, (x + w1, 150), "?!", 144, (255, 60, 60), stroke=10)
        d.rectangle([420, 330, 860, 396], fill=(0, 0, 0))
        d.text((640, 338), "MINES IN THE DMZ", font=k.font(48), fill=(255, 210, 60), anchor="ma")

    finish(low, text, "out/dmz_mine_thumb.png")


# ---------------------------------------------------------------- 숏츠용 세로 표지 (1080x1920)

VW, VH = 135, 240


def finish_v(low, draw_text, path):
    img = low.resize((VW * S, VH * S), Image.NEAREST)
    d = ImageDraw.Draw(img)
    draw_text(d)
    d.rectangle([60, 80, 460, 168], fill=(255, 200, 60))
    d.rectangle([60, 168, 460, 178], fill=(200, 30, 40))
    d.text((80, 88), "KOWEN NEWS", font=k.font(72), fill=(15, 20, 50))
    img.save(path)
    print("완료:", path)


def ocean_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, VH, (30, 90, 170))
    for y in range(4, VH, 8):
        for x in range(0, VW, 12):
            k.rect(d, x + (y % 12), y, 4, 1, (70, 140, 215))
    d.polygon([(105, 120), (135, 110), (135, 240), (118, 240), (108, 190)], fill=(70, 160, 80))
    k.rect(d, 98, 150, 1, 26, (230, 230, 230))
    k.rect(d, 99, 150, 16, 9, (255, 205, 60))
    k.rect(d, 101, 153, 12, 2, (200, 30, 40))
    k.draw_trump(d, 52, 205, 0.05)
    k.rect(d, 0, 205, 105, 35, (230, 200, 120))   # 해변

    def text(d):
        big_text(d, (540, 300), "ATLANTIC", 112, (255, 255, 255), anchor="ma")
        x0, y0, x1, y1 = d.textbbox((540, 300), "ATLANTIC", font=k.font(112), anchor="ma")
        d.rectangle([x0 - 12, (y0 + y1) // 2 - 7, x1 + 12, (y0 + y1) // 2 + 7], fill=(230, 30, 40))
        big_text(d, (540, 450), "TRUMP", 224, (255, 210, 60), stroke=12, anchor="ma")
        big_text(d, (540, 690), "OCEAN?!", 192, (255, 210, 60), stroke=12, anchor="ma")

    finish_v(low, text, "out/trump_ocean_short_thumb.png")


def diesel_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, VH, (40, 50, 130))
    for i in range(0, 120, 12):
        k.rect(d, 0, i, VW, 6, (190, 30, 40))
    k.draw_flag(d, "ru", 0, 120, VW, 120)
    k.rect(d, 0, 118, VW, 4, (10, 10, 15))
    k.draw_trump(d, 40, 122, 0.05)
    k.draw_person(d, 95, 242, 0, False, k.PUTIN, gesture=False)
    k.rect(d, 14, 196, 22, 44, (230, 230, 235))       # 주유기
    k.rect(d, 17, 200, 16, 8, (20, 20, 30))
    k.rect(d, 16, 214, 18, 6, (250, 200, 40))

    def text(d):
        big_text(d, (540, 230), "3 WEEKS LATER", 96, (255, 255, 255), anchor="ma")
        d.rectangle([60, 1000, 1020, 1240], fill=(0, 0, 0))
        big_text(d, (540, 1010), "SANCTIONS", 128, (255, 255, 255), anchor="ma")
        big_text(d, (540, 1120), "WAIVED?!", 112, (255, 210, 60), stroke=10, anchor="ma")
        d.rectangle([340, 1460, 740, 1530], fill=(200, 30, 40))
        d.text((540, 1468), "DIESEL DEAL", font=k.font(56), fill=(255, 255, 255), anchor="ma")

    finish_v(low, text, "out/russia_diesel_short_thumb.png")


if __name__ == "__main__":
    dmz()
    ocean()
    diesel()
    ocean_short()
    diesel_short()
