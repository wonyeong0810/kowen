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


if __name__ == "__main__":
    ocean()
    diesel()
