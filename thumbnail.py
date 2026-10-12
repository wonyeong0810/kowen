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


def ukraine():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, TW, TH, (12, 16, 40))
    # 왼쪽: 성조기
    for i in range(40, TH, 6):
        k.rect(d, 0, i, 80, 3, (190, 30, 40))      # 빨강 줄
        k.rect(d, 0, i + 3, 80, 3, (200, 200, 208))  # 흰 줄 (살짝 톤 다운)
    k.rect(d, 0, 40, 30, 21, (30, 40, 110))      # 남색 칸 + 흰 별
    for sx in range(2, 28, 4):
        for sy in range(42, 60, 4):
            k.rect(d, sx, sy, 1, 1, (220, 220, 230))
    # 오른쪽: 우크라이나 국기 전체 (파랑 위, 노랑 아래)
    k.rect(d, 80, 40, 80, 25, (30, 90, 200))
    k.rect(d, 80, 65, 80, 25, (250, 210, 30))
    k.rect(d, 79, 40, 2, 50, (10, 10, 15))
    k.draw_char(d, 30, 106, 0.05, False, "trump")
    k.draw_char(d, 130, 104, 0.05, False, "zelensky")
    # 젤렌스키 위 빨간 X
    for i in range(-10, 11):
        k.rect(d, 130 + i, 64 + i, 3, 3, (230, 30, 40))
        k.rect(d, 130 + i, 64 - i, 3, 3, (230, 30, 40))
    k.draw_char(d, 80, 106, 0.05, False, "musk")

    def text(d):
        big_text(d, (640, 135), "NEW PRESIDENT", 112, (255, 255, 255), stroke=10, anchor="ma")
        f = k.font(112)
        w1 = d.textlength("FOR UKRAINE", font=f)
        w2 = d.textlength("?!", font=f)
        x = (1280 - w1 - w2) / 2
        big_text(d, (x, 245), "FOR UKRAINE", 112, (255, 210, 60), stroke=10)
        big_text(d, (x + w1, 245), "?!", 112, (255, 60, 60), stroke=10)

    finish(low, text, "out/ukraine_new_leader_thumb.png")


def _two_line(d, y1, t1, c1, y2, t2, c2, size=112, cx=640):
    big_text(d, (cx, y1), t1, size, c1, stroke=10, anchor="ma")
    big_text(d, (cx, y2), t2, size, c2, stroke=10, anchor="ma")


def _fr_flag(d, x, y, w, h):
    k.rect(d, x, y, w // 3, h, (30, 60, 160))
    k.rect(d, x + w // 3, y, w // 3, h, (225, 225, 230))
    k.rect(d, x + 2 * w // 3, y, w - 2 * w // 3, h, (215, 40, 50))


def _crowd(d, y0, w, seed=3):
    import random
    rnd = random.Random(seed)
    for row, y in enumerate((y0, y0 + 8)):
        x = -4 + row * 5
        while x < w + 6:
            k.rect(d, x, y + 6, 9, 20, rnd.choice([(200, 60, 60), (60, 110, 180), (220, 200, 60), (90, 160, 90)]))
            k.rect(d, x + 1, y, 7, 7, rnd.choice(k.SKINS))
            k.rect(d, x + 1, y - 1, 7, 2, rnd.choice(k.HAIRS))
            if rnd.random() < 0.3:
                k.rect(d, x + 4, y - 12, 1, 12, (120, 90, 60))
                k.rect(d, x - 2, y - 19, 14, 8, (245, 245, 240))
                k.rect(d, x, y - 17, 10, 2, (200, 30, 40))
            x += 10 + rnd.randint(0, 2)


def france():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, TW, 44, (12, 16, 40))
    _fr_flag(d, 0, 44, TW, 46)
    _crowd(d, 66, TW)
    k.draw_char(d, 26, 92, 0.05, False, "student")
    k.draw_char(d, 136, 92, 0.05, False, "riot_police")

    def text(d):
        _two_line(d, 135, "260,000 STUDENTS", (255, 255, 255), 245, "REVOLT", (255, 210, 60))

    finish(low, text, "out/france_students_thumb.png")


def _night_launchers(d, w, y_ground, h):
    k.vgrad(d, 0, y_ground, (15, 20, 45), (70, 50, 70))
    k.rect(d, 0, y_ground, w, h - y_ground, (150, 130, 95))
    for lx in (20, 52):
        k.rect(d, lx - 10, y_ground - 8, 26, 8, (90, 95, 70))
        d.polygon([(lx - 6, y_ground - 8), (lx + 12, y_ground - 22), (lx + 16, y_ground - 18), (lx - 2, y_ground - 6)],
                  fill=(110, 115, 85))
    ex, ey = int(w * 0.58), max(14, y_ground - 24)        # 요격 폭발 (제목 띠 아래)
    d.line([(64, y_ground - 22), (ex, ey)], fill=(220, 220, 200))
    for r, c in ((9, (255, 140, 40)), (5, (255, 220, 120))):
        d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=c)


def saudi():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    _night_launchers(d, TW, 80, TH)
    k.rect(d, 0, 0, TW, 40, (12, 16, 40))
    k.draw_char(d, 132, 94, 0.05, False, "mbs")

    def text(d):
        _two_line(d, 135, "SAUDI MISSILES", (255, 255, 255), 245, "RUNNING OUT", (255, 80, 60))

    finish(low, text, "out/saudi_air_defense_thumb.png")


def _biohazard(d, cx, cy, r):
    d.polygon([(cx, cy - r), (cx + r, cy + r * 3 // 4), (cx - r, cy + r * 3 // 4)], fill=(250, 200, 40))
    d.ellipse([cx - r // 3, cy - r // 6, cx + r // 3, cy + r // 2], outline=(20, 20, 20), width=2)


def plague():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, TW, TH, (12, 16, 40))
    k.draw_flag(d, "ru", 0, 44, TW, 46)
    for i in range(0, TW, 8):                          # 격리 테이프
        k.rect(d, i, 82, 8, 4, (250, 210, 40) if (i // 8) % 2 == 0 else (20, 20, 20))
    k.draw_char(d, 34, 96, 0.05, False, "hazmat")
    _biohazard(d, 128, 62, 14)

    def text(d):
        _two_line(d, 135, "PLAGUE SCARE", (255, 255, 255), 245, "IN RUSSIA", (255, 80, 60))

    finish(low, text, "out/russia_plague_thumb.png")


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


def dmz_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.vgrad(d, 0, 150, (60, 20, 20), (160, 70, 30))
    d.polygon([(0, 150), (30, 120), (70, 140), (110, 115), (135, 135), (135, 150)], fill=(50, 40, 35))
    k.rect(d, 0, 150, VW, 90, (80, 95, 55))
    for r, c in ((40, (255, 120, 40)), (28, (255, 190, 80)), (14, (255, 245, 200))):
        d.ellipse([67 - r, 150 - r // 1.4, 67 + r, 150 + r // 1.4], fill=c)
    k.rect(d, 0, 160, VW, 1, (160, 160, 160))
    for fx in range(0, VW, 10):
        k.rect(d, fx, 146, 1, 18, (110, 110, 110))
    for fx in range(0, VW, 3):
        k.rect(d, fx, 152 + (fx // 3) % 2, 2, 1, (180, 180, 180))
    d.polygon([(57, 196), (77, 196), (67, 180)], fill=(220, 30, 30))
    k.rect(d, 66, 186, 2, 5, (255, 255, 255))
    k.rect(d, 66, 192, 2, 2, (255, 255, 255))
    k.draw_char(d, 28, 242, 0.05, False, "soldier")
    k.draw_char(d, 106, 242, 0.05, False, "kim_jong_un")

    def text(d):
        big_text(d, (540, 300), "KOREAN", 208, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 520), "WAR 2?!", 208, (255, 70, 60), stroke=12, anchor="ma")
        d.rectangle([180, 800, 900, 880], fill=(0, 0, 0))
        d.text((540, 810), "MINES IN THE DMZ", font=k.font(64), fill=(255, 210, 60), anchor="ma")

    finish_v(low, text, "out/dmz_mine_short_thumb.png")


def ukraine_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, 120, (12, 16, 40))
    k.draw_flag(d, "ua", 0, 120, VW, 120)
    k.draw_char(d, 34, 182, 0.05, False, "trump")
    k.draw_char(d, 101, 182, 0.05, False, "zelensky")
    for i in range(-10, 11):
        k.rect(d, 101 + i, 142 + i, 3, 3, (230, 30, 40))
        k.rect(d, 101 + i, 142 - i, 3, 3, (230, 30, 40))
    k.draw_char(d, 67, 242, 0.05, False, "musk")

    def text(d):
        big_text(d, (540, 260), "TRUMP WANTS", 120, (255, 255, 255), stroke=10, anchor="ma")
        big_text(d, (540, 420), "ZELENSKY", 160, (255, 210, 60), stroke=12, anchor="ma")
        big_text(d, (540, 610), "OUT?!", 200, (255, 60, 60), stroke=12, anchor="ma")

    finish_v(low, text, "out/ukraine_new_leader_short_thumb.png")


def france_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, 120, (12, 16, 40))
    _fr_flag(d, 0, 120, VW, 120)
    _crowd(d, 150, VW)
    k.draw_char(d, 34, 242, 0.05, False, "student")
    k.draw_char(d, 102, 242, 0.05, False, "riot_police")

    def text(d):
        big_text(d, (540, 300), "260,000", 200, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 540), "STUDENTS", 160, (255, 210, 60), stroke=12, anchor="ma")
        big_text(d, (540, 730), "REVOLT", 120, (255, 80, 60), stroke=10, anchor="ma")

    finish_v(low, text, "out/france_students_short_thumb.png")


def saudi_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, VH, (12, 16, 40))
    sub = Image.new("RGB", (VW, 120))
    _night_launchers(ImageDraw.Draw(sub), VW, 90, 120)
    low.paste(sub, (0, 120))
    k.draw_char(d, 100, 242, 0.05, False, "mbs")

    def text(d):
        big_text(d, (540, 260), "SAUDI", 200, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 470), "MISSILES", 160, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 660), "RUNNING OUT", 96, (255, 80, 60), stroke=10, anchor="ma")

    finish_v(low, text, "out/saudi_air_defense_short_thumb.png")


def plague_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, VH, (12, 16, 40))
    k.draw_flag(d, "ru", 0, 120, VW, 120)
    for i in range(0, VW, 8):
        k.rect(d, i, 176, 8, 4, (250, 210, 40) if (i // 8) % 2 == 0 else (20, 20, 20))
    k.draw_char(d, 40, 242, 0.05, False, "hazmat")
    _biohazard(d, 100, 150, 18)

    def text(d):
        big_text(d, (540, 260), "PLAGUE", 220, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 480), "SCARE", 200, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 700), "IN RUSSIA", 140, (255, 80, 60), stroke=12, anchor="ma")

    finish_v(low, text, "out/russia_plague_short_thumb.png")


def _pylon(d, cx, by, h, fire=True):
    c = (150, 155, 165)
    for i in range(h):                                 # 송전탑 다리
        w = max(1, (h - i) // 4)
        k.rect(d, cx - w, by - i, 1, 1, c)
        k.rect(d, cx + w, by - i, 1, 1, c)
    for y in (by - h * 2 // 3, by - h + 3):           # 팔
        k.rect(d, cx - 9, y, 19, 1, c)
    k.rect(d, cx, by - h, 1, h, c)
    if fire:
        for i, col in enumerate(((230, 60, 30), (250, 140, 30), (255, 220, 80))):
            k.rect(d, cx - 6 + i * 2, by - h - 10 + i * 3, 12 - i * 4, 10 - i * 3, col)


def ceasefire():
    low = Image.new("RGB", (TW, TH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, TW, TH, (12, 16, 40))
    k.draw_flag(d, "ua", 60, 46, 50, 44)
    k.draw_flag(d, "ru", 110, 46, 50, 44)
    k.rect(d, 109, 46, 2, 44, (10, 10, 15))
    _pylon(d, 85, 90, 28)
    _pylon(d, 140, 90, 28)
    k.draw_char(d, 30, 106, 0.05, False, "trump")

    def text(d):
        _two_line(d, 135, "ENERGY CEASEFIRE", (255, 210, 60), 245, "RUSSIA SILENT", (255, 80, 60), size=104)

    finish(low, text, "out/energy_ceasefire_thumb.png")


def ceasefire_short():
    low = Image.new("RGB", (VW, VH))
    d = ImageDraw.Draw(low)
    k.rect(d, 0, 0, VW, VH, (12, 16, 40))
    k.draw_flag(d, "ua", 0, 130, VW // 2, 110)
    k.draw_flag(d, "ru", VW // 2, 130, VW - VW // 2, 110)
    _pylon(d, 100, 200, 40)
    k.draw_char(d, 40, 242, 0.05, False, "trump")

    def text(d):
        big_text(d, (540, 260), "ENERGY", 200, (255, 210, 60), stroke=12, anchor="ma")
        big_text(d, (540, 470), "CEASEFIRE", 160, (255, 255, 255), stroke=12, anchor="ma")
        big_text(d, (540, 660), "RUSSIA SILENT", 110, (255, 80, 60), stroke=12, anchor="ma")

    finish_v(low, text, "out/energy_ceasefire_short_thumb.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    dmz()
    ocean()
    diesel()
    ocean_short()
    diesel_short()
    dmz_short()
    ukraine()
    ukraine_short()
    france()
    saudi()
    plague()
    france_short()
    saudi_short()
    plague_short()
