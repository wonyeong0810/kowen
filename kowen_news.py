#!/usr/bin/env python3
"""KOWEN News - 픽셀 캐릭터 뉴스 영상 생성기.

사용법:
    python3 kowen_news.py episodes/trump_ocean.json

320x180 저해상도 캔버스에 픽셀 아트를 그린 뒤 4배 확대(1280x720)하고,
그 위에 자막/로고/티커를 픽셀 폰트(Unifont)로 얹는다.
효과음(말소리 삑삑, 오프닝 징글, 함성)은 직접 합성해서 ffmpeg로 합친다.
"""
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 320, 180          # 픽셀 아트 캔버스
SCALE = 4                # 1280x720 출력
FW, FH = W * SCALE, H * SCALE
FPS = 24
SR = 44100               # 오디오 샘플레이트
TYPE_SPEED = 16          # 자막 타자 속도 (글자/초)

FONT_PATH = "/usr/share/fonts/opentype/unifont/unifont.otf"
_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_PATH, size)
    return _fonts[size]


def rect(d, x, y, w, h, c):
    if w > 0 and h > 0:
        d.rectangle([x, y, x + w - 1, y + h - 1], fill=c)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vgrad(d, y0, y1, c0, c1):
    for y in range(y0, y1):
        d.line([(0, y), (W, y)], fill=lerp(c0, c1, (y - y0) / max(1, y1 - y0)))


# ---------------------------------------------------------------- 캐릭터

def draw_anchor(d, cx, by, t, talking):
    """뉴스룸 남자 앵커 (책상 뒤 상반신). by = 책상 윗면 y."""
    navy, shirt, tie = (30, 40, 80), (235, 235, 240), (190, 30, 40)
    skin, hair, dark = (240, 195, 160), (40, 28, 20), (20, 20, 25)
    # 몸통 / 수트
    rect(d, cx - 18, by - 24, 36, 26, navy)
    rect(d, cx - 20, by - 18, 2, 20, navy)
    rect(d, cx + 18, by - 18, 2, 20, navy)
    rect(d, cx - 5, by - 24, 10, 10, shirt)
    rect(d, cx - 3, by - 16, 6, 4, shirt)
    rect(d, cx - 1, by - 23, 3, 16, tie)
    rect(d, cx - 2, by - 24, 5, 2, tie)
    # 목 / 머리
    rect(d, cx - 3, by - 28, 7, 5, skin)
    rect(d, cx - 8, by - 46, 17, 19, skin)
    rect(d, cx - 10, by - 39, 2, 5, skin)
    rect(d, cx + 9, by - 39, 2, 5, skin)
    rect(d, cx - 9, by - 49, 19, 6, hair)
    rect(d, cx - 9, by - 44, 3, 5, hair)
    rect(d, cx + 7, by - 44, 3, 3, hair)
    rect(d, cx - 4, by - 50, 12, 2, hair)
    # 눈 (가끔 깜빡)
    blink = (t % 3.7) < 0.12
    for ex in (cx - 5, cx + 3):
        rect(d, ex - 1, by - 41, 4, 1, hair)  # 눈썹
        if blink:
            rect(d, ex, by - 37, 3, 1, dark)
        else:
            rect(d, ex, by - 38, 2, 3, dark)
    rect(d, cx, by - 35, 2, 3, (220, 170, 135))  # 코
    # 입
    if talking and int(t * 9) % 2 == 0:
        rect(d, cx - 3, by - 31, 7, 3, (120, 30, 40))
        rect(d, cx - 2, by - 31, 5, 1, (250, 250, 250))
    else:
        rect(d, cx - 3, by - 30, 7, 1, (150, 80, 70))
    # 손 제스처
    if talking and int(t * 1.3) % 3 == 1:
        bob = int(math.sin(t * 10) * 2)
        rect(d, cx + 20, by - 14 + bob, 4, 12, navy)
        rect(d, cx + 19, by - 18 + bob, 6, 5, skin)


def draw_trump(d, cx, by, t, scale=1):
    """유세 중인 트럼프 풍 캐릭터. by = 발 기준이 아닌 연단 윗면 y."""
    s = scale
    navy, shirt, tie = (25, 35, 75), (240, 240, 245), (210, 25, 35)
    skin, hair, dark = (240, 160, 95), (250, 205, 70), (25, 20, 20)

    def r(x, y, w, h, c):
        rect(d, cx + x * s, by + y * s, w * s, h * s, c)

    # 몸통
    r(-20, -26, 40, 30, navy)
    r(-5, -26, 10, 12, shirt)
    r(-2, -25, 5, 24, tie)
    r(-3, -26, 7, 2, tie)
    # 팔 - 오른팔: 가리키기 / 주먹 흔들기 반복
    phase = int(t * 2.2) % 4
    if phase in (0, 2):     # 하늘 찌르기
        up = 2 if int(t * 8) % 2 else 0
        r(20, -40 - up, 6, 22, navy)
        r(20, -46 - up, 6, 6, skin)
        r(22, -50 - up, 2, 4, skin)
    elif phase == 1:        # 옆으로 손짓
        r(20, -22, 14, 6, navy)
        r(34, -24, 6, 8, skin)
    else:                   # 아코디언 손동작
        w = 4 + int(abs(math.sin(t * 6)) * 6)
        r(14, -16, 6, 6, navy)
        r(14 - w, -18, 5, 6, skin)
        r(-19 + w, -18, 5, 6, skin)
    if phase != 3:
        r(-26, -20, 6, 16, navy)
        r(-26, -6, 6, 5, skin)
    # 머리
    r(-3, -30, 7, 5, skin)
    r(-9, -50, 19, 21, skin)
    r(-11, -43, 2, 5, skin)
    r(10, -43, 2, 5, skin)
    # 머리카락 (스윕)
    r(-11, -54, 22, 6, hair)
    r(-12, -50, 4, 6, hair)
    r(8, -50, 5, 4, hair)
    r(-6, -56, 20, 3, hair)
    r(12, -55, 4, 3, hair)
    # 찡그린 눈썹 + 눈
    for ex in (-6, 3):
        r(ex - 1, -45, 5, 1, (200, 150, 60))
        r(ex, -42, 3, 2, dark)
    r(0, -39, 2, 3, (220, 140, 80))
    # 입: 'O' 모양 / 다문 입 반복
    m = int(t * 7) % 3
    if m == 0:
        r(-2, -34, 5, 4, (110, 25, 30))
    elif m == 1:
        r(-3, -33, 7, 2, (110, 25, 30))
    else:
        r(-2, -33, 5, 1, (160, 80, 60))


CROWD_SKIN = [(240, 195, 160), (200, 140, 100), (130, 85, 60), (250, 215, 185)]
CROWD_HAIR = [(40, 28, 20), (120, 80, 40), (230, 200, 120), (90, 90, 90), (20, 20, 20)]
CROWD_TOP = [(200, 30, 40), (220, 220, 230), (40, 60, 140), (200, 30, 40)]


def draw_crowd(d, t, seed=7, cheer=1.0):
    rnd = random.Random(seed)
    for row, y in enumerate((150, 160, 170)):
        x = -4 + row * 5
        while x < W + 10:
            skin, hair = rnd.choice(CROWD_SKIN), rnd.choice(CROWD_HAIR)
            top = rnd.choice(CROWD_TOP)
            ph = rnd.random() * 6.28
            bob = int(abs(math.sin(t * 6 * cheer + ph)) * 4 * cheer)
            yy = y - bob
            rect(d, x, yy + 9, 12, 20, top)
            rect(d, x + 2, yy, 8, 9, skin)
            if rnd.random() < 0.4:  # 빨간 모자
                rect(d, x + 1, yy - 1, 10, 4, (200, 30, 40))
                rect(d, x + 8, yy + 2, 4, 1, (200, 30, 40))
            else:
                rect(d, x + 1, yy - 1, 10, 3, hair)
            if rnd.random() < 0.25:  # 손 들고 환호
                rect(d, x + 11, yy - 6 - bob, 3, 14, skin)
            if rnd.random() < 0.12:  # 피켓
                rect(d, x + 4, yy - 16, 1, 14, (120, 90, 60))
                rect(d, x - 3, yy - 26, 16, 10, (245, 245, 245))
                rect(d, x - 1, yy - 23, 12, 2, (200, 30, 40))
                rect(d, x - 1, yy - 20, 8, 2, (40, 60, 140))
            x += 13 + rnd.randint(0, 3)


def draw_us_flag(d, x, y, t, w=34, h=20):
    for i in range(7):
        c = (200, 30, 40) if i % 2 == 0 else (245, 245, 245)
        for col in range(w):
            wave_ = int(math.sin(t * 4 + col * 0.25) * 1.5)
            rect(d, x + col, y + i * h // 7 + wave_, 1, h // 7 + 1, c)
    rect(d, x, y + int(math.sin(t * 4) * 1.5), w * 2 // 5, h * 4 // 7, (40, 50, 130))
    for sx in range(x + 2, x + w * 2 // 5 - 1, 3):
        for sy in range(y + 2, y + h * 4 // 7 - 1, 3):
            rect(d, sx, sy + int(math.sin(t * 4) * 1.5), 1, 1, (255, 255, 255))
    rect(d, x - 2, y - 2, 2, h + 40, (180, 180, 190))


# ---------------------------------------------------------------- 장면

class Scene:
    def __init__(self, spec):
        self.spec = spec
        self.lines = spec.get("lines", [])
        self.duration = spec.get("duration") or sum(l["dur"] for l in self.lines)
        # 자막 타임라인
        self.timeline, acc = [], 0.0
        for l in self.lines:
            self.timeline.append((acc, acc + l["dur"], l))
            acc += l["dur"]

    def line_at(self, t):
        for s, e, l in self.timeline:
            if s <= t < e:
                return s, l
        return None, None

    def speaking(self, t, who):
        s, l = self.line_at(t)
        return l is not None and l["speaker"] == who and (t - s) * TYPE_SPEED < len(l["text"]) + 4

    def draw(self, d, t):
        raise NotImplementedError

    def overlay(self, img, d, t):
        pass


class TitleScene(Scene):
    def draw(self, d, t):
        rect(d, 0, 0, W, H, (8, 10, 25))
        # 배경 방사선
        for i in range(24):
            a = i / 24 * 6.283 + t * 0.4
            c = (20, 30, 70) if i % 2 else (12, 16, 40)
            d.polygon([(160, 80), (160 + math.cos(a) * 400, 80 + math.sin(a) * 400),
                       (160 + math.cos(a + 0.26) * 400, 80 + math.sin(a + 0.26) * 400)], fill=c)
        # 회전하는 픽셀 지구본
        cx, cy, R = 160, 70, 26
        for y in range(-R, R + 1):
            for x in range(-R, R + 1):
                if x * x + y * y <= R * R:
                    lon = math.asin(max(-1, min(1, x / math.sqrt(max(1, R * R - y * y))))) + t * 1.5
                    land = math.sin(lon * 3) + math.cos(y / 7.0 + lon) > 0.9
                    c = (60, 170, 80) if land else (40, 110, 210)
                    if x * x + y * y > (R - 2) ** 2:
                        c = (230, 230, 255)
                    rect(d, cx + x, cy + y, 1, 1, c)

    def overlay(self, img, d, t):
        k = min(1.0, t / 0.6)
        y = int(420 - (1 - k) * 60)
        txt = "KOWEN NEWS"
        f = font(96)
        tw = d.textlength(txt, font=f)
        x = (FW - tw) / 2
        for ox, oy in ((6, 6),):
            d.text((x + ox, y + oy), txt, font=f, fill=(0, 0, 0))
        d.text((x, y), txt, font=f, fill=(255, 210, 60))
        if t > 1.0:
            sub = "픽셀로 보는 오늘의 뉴스"
            d.text(((FW - d.textlength(sub, font=font(32))) / 2, y + 120), sub,
                   font=font(32), fill=(230, 230, 255))


class AnchorScene(Scene):
    def draw(self, d, t):
        vgrad(d, 0, 120, (20, 35, 80), (45, 70, 140))
        # 뒤쪽 창문 야경
        rnd = random.Random(3)
        for bx in range(0, 180, 22):
            bh = rnd.randint(25, 60)
            rect(d, bx, 120 - bh, 20, bh, (25, 30, 60))
            for wy in range(120 - bh + 4, 116, 6):
                for wx in range(bx + 3, bx + 18, 5):
                    if rnd.random() < 0.5:
                        rect(d, wx, wy, 2, 3, (255, 220, 120))
        # 창틀
        for fx in range(0, 190, 46):
            rect(d, fx, 0, 3, 120, (15, 20, 45))
        rect(d, 0, 0, 190, 6, (15, 20, 45))
        # 뒤 대형 스크린
        rect(d, 190, 14, 122, 86, (10, 10, 15))
        rect(d, 194, 18, 114, 78, (30, 90, 170))
        # 스크린 안: 바다 + 물음표
        for i in range(0, 114, 6):
            yy = 60 + int(math.sin(t * 3 + i * 0.3) * 2)
            rect(d, 194 + i, yy, 6, 2, (90, 160, 230))
        rect(d, 196, 30, 22, 50, (60, 160, 80))
        rect(d, 286, 26, 20, 56, (60, 160, 80))
        if int(t * 2) % 2 == 0:
            rect(d, 247, 32, 10, 3, (255, 220, 60))
            rect(d, 255, 35, 3, 6, (255, 220, 60))
            rect(d, 250, 41, 6, 3, (255, 220, 60))
            rect(d, 250, 47, 4, 3, (255, 220, 60))
        # 앵커
        draw_anchor(d, 110, 106, t, self.speaking(t, "앵커"))
        # 책상
        rect(d, 0, 120, W, 60, (20, 25, 50))
        rect(d, 30, 102, 210, 6, (200, 205, 215))
        rect(d, 40, 108, 190, 72, (35, 45, 90))
        rect(d, 40, 108, 190, 3, (255, 200, 60))
        # 서류
        rect(d, 92, 99, 22, 4, (250, 250, 250))

    def overlay(self, img, d, t):
        # 책상 앞 로고
        d.text((320, 460), "KOWEN NEWS", font=font(48), fill=(255, 210, 60))
        # LIVE 칩
        rect(d, 32, 28, 96, 40, (210, 30, 40))
        if int(t * 2) % 2 == 0:
            d.ellipse([42, 40, 58, 56], fill=(255, 255, 255))
        d.text((64, 32), "LIVE", font=font(32), fill=(255, 255, 255))
        head = self.spec.get("headline")
        if head:
            rect(d, 784, 412, 456, 44, (0, 0, 0))
            d.text((796, 418), head, font=font(32), fill=(255, 230, 120))


class RallyScene(Scene):
    def draw(self, d, t):
        vgrad(d, 0, 110, (10, 10, 30), (40, 20, 60))
        # 스포트라이트
        for i, bx in enumerate((40, 280)):
            sw = math.sin(t * 0.8 + i * 2) * 30
            d.polygon([(bx, 0), (160 + sw - 25, 120), (160 + sw + 25, 120)], fill=(70, 60, 100))
        # 성조기 배경
        draw_us_flag(d, 18, 30, t)
        draw_us_flag(d, 266, 30, t + 1)
        # 무대
        rect(d, 0, 110, W, 40, (60, 30, 30))
        rect(d, 0, 110, W, 3, (220, 180, 80))
        # 트럼프 + 연단
        draw_trump(d, 160, 112, t)
        rect(d, 136, 104, 48, 46, (30, 40, 90))
        rect(d, 132, 100, 56, 6, (90, 70, 50))
        rect(d, 158, 92, 2, 10, (40, 40, 40))       # 마이크
        rect(d, 156, 90, 6, 4, (70, 70, 70))
        # 관중
        draw_crowd(d, t)

    def overlay(self, img, d, t):
        banner = self.spec.get("banner", "")
        f = font(48)
        tw = d.textlength(banner, font=f)
        rect(d, int((FW - tw) / 2) - 24, 28, int(tw) + 48, 64, (200, 30, 40))
        d.text(((FW - tw) / 2, 36), banner, font=f, fill=(255, 255, 255))
        pod = self.spec.get("podium", "")
        pf = font(32)
        d.text(((FW - d.textlength(pod, font=pf)) / 2, 438), pod, font=pf, fill=(255, 255, 255))
        rect(d, 1080, 40, 160, 40, (0, 0, 0))
        d.text((1096, 44), "현장 재연", font=font(32), fill=(255, 230, 120))


class MapScene(Scene):
    AMERICAS = [(0, 10), (60, 6), (78, 30), (66, 60), (52, 74), (70, 96), (84, 120),
                (74, 150), (58, 180), (0, 180)]
    EURAFRICA = [(320, 4), (250, 8), (236, 30), (250, 50), (232, 64), (226, 90),
                 (246, 112), (262, 150), (276, 180), (320, 180)]

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (30, 90, 170))
        for y in range(6, H, 12):
            for x in range(0, W, 16):
                off = int((t * 10 + y) % 16)
                rect(d, (x + off) % W, y, 5, 1, (70, 140, 215))
        d.polygon(self.AMERICAS, fill=(70, 160, 80))
        d.polygon(self.EURAFRICA, fill=(70, 160, 80))
        # 좌표선
        for gx in range(0, W, 40):
            for gy in range(0, H, 4):
                rect(d, gx, gy, 1, 1, (60, 120, 200))
        # 깃발 떨어지기 (중반부)
        p = self.progress(t)
        if p > 0.45:
            k = min(1.0, (p - 0.45) / 0.08)
            fy = int(-40 + k * 120)
            rect(d, 158, fy, 2, 36, (230, 230, 230))
            wave_ = int(math.sin(t * 6) * 1)
            rect(d, 160, fy + wave_, 22, 12, (255, 205, 60))
            rect(d, 163, fy + 3 + wave_, 16, 2, (200, 30, 40))
            if k >= 1.0:  # 물결 퍼짐
                rr = int(((t * 20) % 16))
                d.ellipse([159 - rr * 2, 115 - rr // 2, 159 + rr * 2, 115 + rr // 2],
                          outline=(200, 230, 255))
        # 미니 트럼프 (왼쪽 해안에서 손짓)
        draw_trump(d, 56, 124, t, scale=1)

    def progress(self, t):
        return t / self.duration

    def overlay(self, img, d, t):
        p = self.progress(t)
        f = font(48)
        old = f'{self.spec["old"]} · {self.spec["old_ko"]}'
        tw = d.textlength(old, font=f)
        x, y = (FW - tw) / 2 + 60, 40
        d.text((x + 4, y + 4), old, font=f, fill=(0, 0, 0))
        d.text((x, y), old, font=f, fill=(255, 255, 255))
        if p > 0.3:
            k = min(1.0, (p - 0.3) / 0.08)
            rect(d, int(x) - 8, y + 24, int((tw + 16) * k), 8, (230, 30, 40))
        if p > 0.55 and (p > 0.62 or int(t * 6) % 2 == 0):
            new = f'{self.spec["new_ko"]} ({self.spec["new"]})'
            f2 = font(64)
            tw2 = d.textlength(new, font=f2)
            nx = (FW - tw2) / 2 + 60
            d.text((nx + 5, 120 + 5), new, font=f2, fill=(0, 0, 0))
            d.text((nx, 120), new, font=f2, fill=(255, 210, 60))
            d.text((nx + tw2 + 10, 110), "?!", font=f2, fill=(255, 80, 80))


SCENES = {"title": TitleScene, "anchor": AnchorScene, "rally": RallyScene, "map": MapScene}


# ---------------------------------------------------------------- 자막 / 티커

def wrap(d, text, f, maxw):
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > maxw:
            cut = cur.rfind(" ")
            if cut > 0:
                lines.append(cur[:cut])
                cur = cur[cut + 1:] + ch
            else:
                lines.append(cur)
                cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


SPEAKER_STYLE = {
    "앵커": ("앵커", (40, 70, 160)),
    "기자": ("기자 (목소리)", (200, 30, 40)),
}


def draw_subtitle(d, scene, t):
    s, l = scene.line_at(t)
    if l is None:
        return
    shown = l["text"][: int((t - s) * TYPE_SPEED)]
    f = font(32)
    rows = wrap(d, l["text"], f, FW - 160)
    # 전체 텍스트로 줄 위치 고정, 타이핑된 만큼만 출력
    box_h = 20 + len(rows) * 40
    y0 = 652 - box_h
    rect(d, 48, y0, FW - 96, box_h, (0, 0, 0))
    rect(d, 48, y0, 6, box_h, SPEAKER_STYLE.get(l["speaker"], (l["speaker"], (90, 90, 90)))[1])
    label, col = SPEAKER_STYLE.get(l["speaker"], (l["speaker"], (90, 90, 90)))
    lw = d.textlength(label, font=f)
    rect(d, 48, y0 - 40, int(lw) + 24, 40, col)
    d.text((60, y0 - 38), label, font=f, fill=(255, 255, 255))
    remain = len(shown)
    for i, row in enumerate(rows):
        part = row[:max(0, remain)]
        remain -= len(row) + (1 if l["text"][len(row):len(row) + 1] == " " else 0)
        d.text((76, y0 + 10 + i * 40), part, font=f, fill=(255, 255, 255))


def draw_ticker(d, ticker, gt):
    rect(d, 0, 664, FW, 56, (15, 20, 50))
    rect(d, 0, 664, 180, 56, (255, 200, 60))
    d.text((14, 676), "KOWEN", font=font(32), fill=(15, 20, 50))
    text = "   ◆   ".join(ticker) + "   ◆   "
    f = font(32)
    tw = d.textlength(text, font=f)
    x = 200 - (gt * 120) % tw
    while x < FW:
        d.text((x, 676), text, font=f, fill=(255, 255, 255))
        x += tw
    rect(d, 0, 664, 180, 56, (255, 200, 60))
    d.text((14, 676), "KOWEN", font=font(32), fill=(15, 20, 50))


# ---------------------------------------------------------------- 오디오

def square(freq, dur, vol):
    n = int(SR * dur)
    tt = np.arange(n) / SR
    w = np.sign(np.sin(2 * np.pi * freq * tt)) * vol
    env = np.minimum(1, np.minimum(tt / 0.005, (dur - tt) / 0.02).clip(0))
    return w * env


VOICE = {"앵커": 210, "기자": 320}


def build_audio(scenes, total):
    buf = np.zeros(int(SR * (total + 1)))
    rnd = random.Random(1)

    def add(start, sig):
        i = int(start * SR)
        buf[i:i + len(sig)] += sig[: len(buf) - i]

    t0 = 0.0
    for sc in scenes:
        if isinstance(sc, TitleScene):  # 오프닝 징글
            notes = [523, 659, 784, 1047, 784, 1047]
            for k, nfreq in enumerate(notes):
                add(t0 + 0.2 + k * 0.16, square(nfreq, 0.15, 0.18))
                add(t0 + 0.2 + k * 0.16, square(nfreq / 2, 0.15, 0.10))
        if isinstance(sc, RallyScene):  # 군중 함성 + 트럼프 웅얼거림
            n = int(SR * sc.duration)
            noise = np.convolve(np.random.default_rng(0).normal(0, 1, n), np.ones(40) / 40, "same")
            swell = 0.5 + 0.5 * np.abs(np.sin(np.arange(n) / SR * 1.3))
            add(t0, noise * 0.12 * swell)
            tt = 0.0
            while tt < sc.duration:
                add(t0 + tt, square(110 + rnd.randint(-15, 25), 0.06, 0.04))
                tt += 0.14
        for s, e, l in sc.timeline:
            base = VOICE.get(l["speaker"], 260)
            for k, ch in enumerate(l["text"]):
                if ch != " " and k % 2 == 0:
                    add(t0 + s + k / TYPE_SPEED, square(base * rnd.uniform(0.9, 1.25), 0.045, 0.08))
        t0 += sc.duration
    buf = np.clip(buf, -1, 1)
    return (buf * 32767).astype(np.int16)


# ---------------------------------------------------------------- 렌더

def render(ep_path):
    with open(ep_path, encoding="utf-8") as fp:
        ep = json.load(fp)
    scenes = [SCENES[s["type"]](s) for s in ep["scenes"]]
    total = sum(s.duration for s in scenes)
    out = ep.get("output", "out/" + os.path.splitext(os.path.basename(ep_path))[0] + ".mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    tmp = tempfile.mkdtemp()
    wav_path = os.path.join(tmp, "audio.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(build_audio(scenes, total).tobytes())

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}", "-r", str(FPS), "-i", "-",
         "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", "-b:a", "128k", "-shortest", out],
        stdin=subprocess.PIPE)

    nframes = int(total * FPS)
    t0, idx = 0.0, 0
    for fi in range(nframes):
        gt = fi / FPS
        while idx < len(scenes) - 1 and gt >= t0 + scenes[idx].duration:
            t0 += scenes[idx].duration
            idx += 1
        sc, t = scenes[idx], gt - t0
        low = Image.new("RGB", (W, H))
        sc.draw(ImageDraw.Draw(low), t)
        img = low.resize((FW, FH), Image.NEAREST)
        d = ImageDraw.Draw(img)
        sc.overlay(img, d, t)
        if not isinstance(sc, TitleScene):
            draw_subtitle(d, sc, t)
            draw_ticker(d, ep.get("ticker", []), gt)
        # 장면 전환 페이드 (픽셀 디졸브 느낌)
        fade = min(t / 0.25, (sc.duration - t) / 0.25, 1.0)
        if fade < 1.0:
            img = Image.blend(Image.new("RGB", img.size), img, max(0.0, fade))
        proc.stdin.write(img.tobytes())
        if fi % (FPS * 5) == 0:
            print(f"  {gt:5.1f}s / {total:.1f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("완료:", out)
    return out


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "episodes/trump_ocean.json")
