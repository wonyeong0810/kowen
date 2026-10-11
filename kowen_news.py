#!/usr/bin/env python3
"""KOWEN News - 픽셀 캐릭터 뉴스 영상 생성기.

사용법:
    python3 kowen_news.py episodes/trump_ocean.json

320x180 저해상도 캔버스에 픽셀 아트를 그린 뒤 확대(1280x720)하고,
그 위에 자막/로고/티커를 픽셀 폰트(Unifont)로 얹는다.
대사는 Supertonic 3 TTS(오프라인, 영어/한국어 남녀 목소리, episode의 "lang")로 읽고,
각 줄의 길이는 실제 음성 길이에 맞춰 자동으로 정해진다.
"""
import hashlib
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
FW, FH = 1280, 720       # 출력 해상도
FPS = 24
SR = 44100               # 오디오 샘플레이트
GAP = 0.15               # 대사 사이 쉼 (초)
ZOOMS = [(4, 320, 180), (5, 256, 144)]   # (배율, 잘라낼 폭, 높이) - 정수 배율이라 픽셀이 고르게 유지됨

FONT_PATH = "/usr/share/fonts/opentype/unifont/unifont.otf"
TTS_DIR = os.environ.get(
    "KOWEN_TTS_MODEL", os.path.expanduser("~/models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"))
if not os.path.isdir(TTS_DIR):
    TTS_DIR = "/home/user/models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
CACHE_DIR = ".cache/tts"

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


def ease(x):
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


# ---------------------------------------------------------------- 카메라

class Cam:
    """저해상도 좌표 -> 출력 좌표 변환 (줌 컷 대응)."""

    def __init__(self, s, ox, oy):
        self.s, self.ox, self.oy = s, ox, oy

    def pt(self, x, y):
        return (x - self.ox) * self.s, (y - self.oy) * self.s

    def text(self, d, x, y, txt, size, fill, center=False, shadow=True):
        f = font(int(size * self.s / 4))
        sx, sy = self.pt(x, y)
        if center:
            sx -= d.textlength(txt, font=f) / 2
        if shadow:
            d.text((sx + self.s, sy + self.s), txt, font=f, fill=(0, 0, 0))
        d.text((sx, sy), txt, font=f, fill=fill)
        return sx, sy, d.textlength(txt, font=f)


# ---------------------------------------------------------------- TTS

class Voice:
    def __init__(self, lang="en"):
        self.tts = None
        self.lang = lang

    def _load(self):
        import sherpa_onnx
        D = TTS_DIR + "/"
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor=D + "duration_predictor.int8.onnx",
                text_encoder=D + "text_encoder.int8.onnx",
                vector_estimator=D + "vector_estimator.int8.onnx",
                vocoder=D + "vocoder.int8.onnx",
                tts_json=D + "tts.json",
                unicode_indexer=D + "unicode_indexer.bin",
                voice_style=D + "voice.bin"),
            num_threads=os.cpu_count() or 4))
        self.tts = sherpa_onnx.OfflineTts(cfg)

    def say(self, text, sid, speed):
        key = hashlib.sha1(f"{self.lang}|{sid}|{speed}|{text}".encode()).hexdigest()
        path = os.path.join(CACHE_DIR, key + ".npy")
        if os.path.exists(path):
            return np.load(path)
        if self.tts is None:
            self._load()
        import sherpa_onnx
        g = sherpa_onnx.GenerationConfig()
        g.sid, g.num_steps, g.speed = sid, 8, speed
        g.extra["lang"] = self.lang
        a = self.tts.generate(text, g)
        x = np.asarray(a.samples, dtype=np.float32)
        if a.sample_rate != SR:
            n = int(len(x) * SR / a.sample_rate)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        # 앞뒤 무음 잘라서 템포 올리기
        idx = np.where(np.abs(x) > 0.01)[0]
        if len(idx):
            x = x[max(0, idx[0] - 400): idx[-1] + 2000]
        x = x / max(1e-6, np.abs(x).max()) * 0.9
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, x)
        return x


# ---------------------------------------------------------------- 캐릭터

SKINS = [(240, 195, 160), (200, 140, 100), (130, 85, 60), (250, 215, 185)]
HAIRS = [(40, 28, 20), (120, 80, 40), (230, 200, 120), (110, 110, 115), (20, 20, 20)]


def draw_person(d, cx, by, t, talking, p, gesture=True):
    """상반신 인물 (앵커/시민/전문가 공용). by = 허리선 y."""
    skin = SKINS[p.get("skin", 0)]
    hair = HAIRS[p.get("hair", 0)]
    top = tuple(p.get("top", (30, 40, 80)))
    dark = (20, 20, 25)
    if p.get("long"):
        rect(d, cx - 10, by - 46, 21, 26, hair)
    rect(d, cx - 18, by - 24, 36, 26, top)
    rect(d, cx - 20, by - 18, 2, 20, top)
    rect(d, cx + 18, by - 18, 2, 20, top)
    if p.get("suit", False):
        rect(d, cx - 5, by - 24, 10, 10, (235, 235, 240))
        rect(d, cx - 3, by - 16, 6, 4, (235, 235, 240))
        rect(d, cx - 1, by - 23, 3, 16, tuple(p.get("tie", (190, 30, 40))))
        rect(d, cx - 2, by - 24, 5, 2, tuple(p.get("tie", (190, 30, 40))))
    else:
        rect(d, cx - 4, by - 24, 8, 3, skin)
    rect(d, cx - 3, by - 28, 7, 5, skin)
    rect(d, cx - 8, by - 46, 17, 19, skin)
    rect(d, cx - 10, by - 39, 2, 5, skin)
    rect(d, cx + 9, by - 39, 2, 5, skin)
    rect(d, cx - 9, by - 49, 19, 6, hair)
    rect(d, cx - 9, by - 44, 3, 5, hair)
    rect(d, cx + 7, by - 44, 3, 3, hair)
    rect(d, cx - 4, by - 50, 12, 2, hair)
    if p.get("long"):
        rect(d, cx - 11, by - 44, 3, 18, hair)
        rect(d, cx + 9, by - 44, 3, 18, hair)
    if p.get("hat"):
        rect(d, cx - 10, by - 52, 21, 6, tuple(p["hat"]))
        rect(d, cx - 12, by - 47, 25, 2, tuple(p["hat"]))
    blink = ((t + cx * 0.37) % 3.7) < 0.12
    for ex in (cx - 5, cx + 3):
        rect(d, ex - 1, by - 41, 4, 1, hair)
        if blink:
            rect(d, ex, by - 37, 3, 1, dark)
        else:
            rect(d, ex, by - 38, 2, 3, dark)
    if p.get("glasses"):
        for ex in (cx - 7, cx + 1):
            d.rectangle([ex, by - 40, ex + 5, by - 35], outline=(30, 30, 30))
        rect(d, cx - 2, by - 38, 3, 1, (30, 30, 30))
    rect(d, cx, by - 35, 2, 3, lerp(skin, (150, 90, 60), 0.3))
    if talking:
        rect(d, cx - 3, by - 31, 7, 3, (120, 30, 40))
        rect(d, cx - 2, by - 31, 5, 1, (250, 250, 250))
    else:
        rect(d, cx - 3, by - 30, 7, 1, (150, 80, 70))
    if gesture and talking and int(t * 1.3 + cx) % 3 == 1:
        bob = int(math.sin(t * 10) * 2)
        rect(d, cx + 20, by - 14 + bob, 4, 12, top)
        rect(d, cx + 19, by - 18 + bob, 6, 5, skin)


ANCHOR = {"skin": 0, "hair": 0, "top": (30, 40, 80), "suit": True}


def draw_trump(d, cx, by, t, talking=True):
    navy, shirt, tie = (25, 35, 75), (240, 240, 245), (210, 25, 35)
    skin, hair, dark = (240, 160, 95), (250, 205, 70), (25, 20, 20)

    def r(x, y, w, h, c):
        rect(d, cx + x, by + y, w, h, c)

    r(-20, -26, 40, 30, navy)
    r(-5, -26, 10, 12, shirt)
    r(-2, -25, 5, 24, tie)
    r(-3, -26, 7, 2, tie)
    phase = int(t * 2.2) % 4
    if phase in (0, 2):
        up = 2 if int(t * 8) % 2 else 0
        r(20, -40 - up, 6, 22, navy)
        r(20, -46 - up, 6, 6, skin)
        r(22, -50 - up, 2, 4, skin)
    elif phase == 1:
        r(20, -22, 14, 6, navy)
        r(34, -24, 6, 8, skin)
    else:
        w = 4 + int(abs(math.sin(t * 6)) * 6)
        r(14, -16, 6, 6, navy)
        r(14 - w, -18, 5, 6, skin)
        r(-19 + w, -18, 5, 6, skin)
    if phase != 3:
        r(-26, -20, 6, 16, navy)
        r(-26, -6, 6, 5, skin)
    r(-3, -30, 7, 5, skin)
    r(-9, -50, 19, 21, skin)
    r(-11, -43, 2, 5, skin)
    r(10, -43, 2, 5, skin)
    r(-11, -54, 22, 6, hair)
    r(-12, -50, 4, 6, hair)
    r(8, -50, 5, 4, hair)
    r(-6, -56, 20, 3, hair)
    r(12, -55, 4, 3, hair)
    for ex in (-6, 3):
        r(ex - 1, -45, 5, 1, (200, 150, 60))
        r(ex, -42, 3, 2, dark)
    r(0, -39, 2, 3, (220, 140, 80))
    m = int(t * 7) % 3 if talking else 2
    if m == 0:
        r(-2, -34, 5, 4, (110, 25, 30))
    elif m == 1:
        r(-3, -33, 7, 2, (110, 25, 30))
    else:
        r(-2, -33, 5, 1, (160, 80, 60))


CROWD_TOP = [(200, 30, 40), (220, 220, 230), (40, 60, 140), (200, 30, 40)]


def draw_crowd(d, t, seed=7, cheer=1.0):
    rnd = random.Random(seed)
    for row, y in enumerate((150, 160, 170)):
        x = -4 + row * 5
        while x < W + 10:
            skin, hair = rnd.choice(SKINS), rnd.choice(HAIRS)
            top = rnd.choice(CROWD_TOP)
            ph = rnd.random() * 6.28
            bob = int(abs(math.sin(t * 6 * cheer + ph)) * 4 * cheer)
            yy = y - bob
            rect(d, x, yy + 9, 12, 20, top)
            rect(d, x + 2, yy, 8, 9, skin)
            if rnd.random() < 0.4:
                rect(d, x + 1, yy - 1, 10, 4, (200, 30, 40))
                rect(d, x + 8, yy + 2, 4, 1, (200, 30, 40))
            else:
                rect(d, x + 1, yy - 1, 10, 3, hair)
            if rnd.random() < 0.25:
                rect(d, x + 11, yy - 6 - bob, 3, 14, skin)
            if rnd.random() < 0.12:
                rect(d, x + 4, yy - 16, 1, 14, (120, 90, 60))
                rect(d, x - 3, yy - 26, 16, 10, (245, 245, 245))
                rect(d, x - 1, yy - 23, 12, 2, (200, 30, 40))
                rect(d, x - 1, yy - 20, 8, 2, (40, 60, 140))
            x += 13 + rnd.randint(0, 3)


def draw_us_flag(d, x, y, t, w=34, h=20):
    for i in range(7):
        c = (200, 30, 40) if i % 2 == 0 else (245, 245, 245)
        for col in range(w):
            wv = int(math.sin(t * 4 + col * 0.25) * 1.5)
            rect(d, x + col, y + i * h // 7 + wv, 1, h // 7 + 1, c)
    rect(d, x, y + int(math.sin(t * 4) * 1.5), w * 2 // 5, h * 4 // 7, (40, 50, 130))
    for sx in range(x + 2, x + w * 2 // 5 - 1, 3):
        for sy in range(y + 2, y + h * 4 // 7 - 1, 3):
            rect(d, sx, sy + int(math.sin(t * 4) * 1.5), 1, 1, (255, 255, 255))
    rect(d, x - 2, y - 2, 2, h + 40, (180, 180, 190))


def draw_flag(d, kind, x, y, w, h):
    if kind == "uk":
        rect(d, x, y, w, h, (30, 40, 120))
        d.line([(x, y), (x + w - 1, y + h - 1)], fill=(240, 240, 240), width=3)
        d.line([(x, y + h - 1), (x + w - 1, y)], fill=(240, 240, 240), width=3)
        rect(d, x, y + h // 2 - 3, w, 6, (240, 240, 240))
        rect(d, x + w // 2 - 3, y, 6, h, (240, 240, 240))
        rect(d, x, y + h // 2 - 2, w, 4, (200, 30, 40))
        rect(d, x + w // 2 - 2, y, 4, h, (200, 30, 40))
    elif kind == "fr":
        rect(d, x, y, w // 3, h, (30, 60, 160))
        rect(d, x + w // 3, y, w // 3, h, (240, 240, 240))
        rect(d, x + 2 * w // 3, y, w - 2 * w // 3, h, (220, 40, 50))
    elif kind == "br":
        rect(d, x, y, w, h, (30, 150, 70))
        d.polygon([(x + w // 2, y + 2), (x + w - 3, y + h // 2), (x + w // 2, y + h - 3), (x + 2, y + h // 2)],
                  fill=(250, 210, 40))
        d.ellipse([x + w // 2 - h // 4, y + h // 4, x + w // 2 + h // 4, y + 3 * h // 4], fill=(30, 50, 140))
    elif kind == "ca":
        rect(d, x, y, w, h, (240, 240, 240))
        rect(d, x, y, w // 4, h, (210, 30, 40))
        rect(d, x + w - w // 4, y, w // 4, h, (210, 30, 40))
        cx, cy = x + w // 2, y + h // 2
        d.polygon([(cx, cy - 7), (cx + 6, cy - 1), (cx + 3, cy + 5), (cx - 3, cy + 5), (cx - 6, cy - 1)],
                  fill=(210, 30, 40))
        rect(d, cx, cy + 5, 1, 3, (210, 30, 40))


# ---------------------------------------------------------------- 장면

class Scene:
    focus = (160, 80)

    def __init__(self, spec, voices, voice_cfg):
        self.spec = spec
        self.timeline, acc = [], 0.0
        for l in spec.get("lines", []):
            vc = voice_cfg.get(l["speaker"], {"sid": 6, "speed": 1.2})
            audio = voices.say(l["text"], vc["sid"], vc.get("speed", 1.2))
            adur = len(audio) / SR
            hop = SR // FPS
            env = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2)) for i in range(0, len(audio), hop)])
            self.timeline.append({"start": acc, "end": acc + adur + GAP, "adur": adur,
                                  "line": l, "audio": audio, "env": env})
            acc += adur + GAP
        self.duration = spec.get("duration") or acc

    def line_at(self, t):
        for i, e in enumerate(self.timeline):
            if e["start"] <= t < e["end"]:
                return i, e
        return -1, None

    def talking(self, t, who):
        _, e = self.line_at(t)
        if e is None or e["line"]["speaker"] != who:
            return False
        k = int((t - e["start"]) * FPS)
        return 0 <= k < len(e["env"]) and e["env"][k] > 0.06 and int(t * 12) % 3 != 2

    def speaker(self, t):
        _, e = self.line_at(t)
        return e["line"]["speaker"] if e else None

    def focus_at(self, t):
        return self.focus

    def draw(self, d, t):
        raise NotImplementedError

    def overlay(self, d, t, cam):
        pass


class TitleScene(Scene):
    zoomable = False

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (8, 10, 25))
        for i in range(24):
            a = i / 24 * 6.283 + t * 0.8
            c = (20, 30, 70) if i % 2 else (12, 16, 40)
            d.polygon([(160, 80), (160 + math.cos(a) * 400, 80 + math.sin(a) * 400),
                       (160 + math.cos(a + 0.26) * 400, 80 + math.sin(a + 0.26) * 400)], fill=c)
        cx, cy, R = 160, 70, 26
        for y in range(-R, R + 1):
            for x in range(-R, R + 1):
                if x * x + y * y <= R * R:
                    lon = math.asin(max(-1, min(1, x / math.sqrt(max(1, R * R - y * y))))) + t * 2
                    land = math.sin(lon * 3) + math.cos(y / 7.0 + lon) > 0.9
                    c = (60, 170, 80) if land else (40, 110, 210)
                    if x * x + y * y > (R - 2) ** 2:
                        c = (230, 230, 255)
                    rect(d, cx + x, cy + y, 1, 1, c)

    def overlay(self, d, t, cam):
        k = ease(t / 0.4)
        y = int(420 - (1 - k) * 80)
        txt = "KOWEN NEWS"
        f = font(96)
        x = (FW - d.textlength(txt, font=f)) / 2
        d.text((x + 6, y + 6), txt, font=f, fill=(0, 0, 0))
        d.text((x, y), txt, font=f, fill=(255, 210, 60))
        if t > 0.7:
            sub = "Today's news, in pixels"
            d.text(((FW - d.textlength(sub, font=font(32))) / 2, y + 120), sub,
                   font=font(32), fill=(230, 230, 255))
        if t > 1.2:
            note = "* Satirical reenactment based on a hypothetical scenario"
            d.text(((FW - d.textlength(note, font=font(16))) / 2, 690), note,
                   font=font(16), fill=(160, 160, 190))


class AnchorScene(Scene):
    focus = (130, 70)

    def draw(self, d, t):
        vgrad(d, 0, 120, (20, 35, 80), (45, 70, 140))
        rnd = random.Random(3)
        for bx in range(0, 180, 22):
            bh = rnd.randint(25, 60)
            rect(d, bx, 120 - bh, 20, bh, (25, 30, 60))
            for wy in range(120 - bh + 4, 116, 6):
                for wx in range(bx + 3, bx + 18, 5):
                    if rnd.random() < 0.5:
                        rect(d, wx, wy, 2, 3, (255, 220, 120))
        for fx in range(0, 190, 46):
            rect(d, fx, 0, 3, 120, (15, 20, 45))
        rect(d, 0, 0, 190, 6, (15, 20, 45))
        rect(d, 190, 14, 122, 86, (10, 10, 15))
        rect(d, 194, 18, 114, 78, (30, 90, 170))
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
        draw_person(d, 110, 106, t, self.talking(t, "Anchor"), ANCHOR)
        rect(d, 0, 120, W, 60, (20, 25, 50))
        rect(d, 30, 102, 210, 6, (200, 205, 215))
        rect(d, 40, 108, 190, 72, (35, 45, 90))
        rect(d, 40, 108, 190, 3, (255, 200, 60))
        rect(d, 92, 99, 22, 4, (250, 250, 250))

    def overlay(self, d, t, cam):
        rect(d, 32, 28, 96, 40, (210, 30, 40))
        if int(t * 2) % 2 == 0:
            d.ellipse([42, 40, 58, 56], fill=(255, 255, 255))
        d.text((64, 32), "LIVE", font=font(32), fill=(255, 255, 255))
        cam.text(d, 135, 116, "KOWEN NEWS", 48, (255, 210, 60), center=True, shadow=False)
        head = self.spec.get("headline")
        if head:
            sx, sy = cam.pt(196, 103)
            f = font(int(32 * cam.s / 4))
            rect(d, int(sx), int(sy), int(d.textlength(head, font=f)) + 24, int(11 * cam.s), (0, 0, 0))
            d.text((sx + 12, sy + cam.s), head, font=f, fill=(255, 230, 120))


class RallyScene(Scene):
    focus = (160, 64)

    def draw(self, d, t):
        vgrad(d, 0, 110, (10, 10, 30), (40, 20, 60))
        for i, bx in enumerate((40, 280)):
            sw = math.sin(t * 0.8 + i * 2) * 30
            d.polygon([(bx, 0), (160 + sw - 25, 120), (160 + sw + 25, 120)], fill=(70, 60, 100))
        draw_us_flag(d, 18, 30, t)
        draw_us_flag(d, 266, 30, t + 1)
        rect(d, 0, 110, W, 40, (60, 30, 30))
        rect(d, 0, 110, W, 3, (220, 180, 80))
        draw_trump(d, 160, 112, t)
        rect(d, 136, 104, 48, 46, (30, 40, 90))
        rect(d, 132, 100, 56, 6, (90, 70, 50))
        rect(d, 158, 92, 2, 10, (40, 40, 40))
        rect(d, 156, 90, 6, 4, (70, 70, 70))
        draw_crowd(d, t)

    def overlay(self, d, t, cam):
        banner = self.spec.get("banner", "")
        f = font(48)
        tw = d.textlength(banner, font=f)
        rect(d, int((FW - tw) / 2) - 24, 28, int(tw) + 48, 64, (200, 30, 40))
        d.text(((FW - tw) / 2, 36), banner, font=f, fill=(255, 255, 255))
        cam.text(d, 160, 112, self.spec.get("podium", ""), 32, (255, 255, 255), center=True, shadow=False)
        rect(d, 1080, 40, 176, 40, (0, 0, 0))
        d.text((1096, 44), "REENACTED", font=font(32), fill=(255, 230, 120))


class MapScene(Scene):
    focus = (160, 60)
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
        for gx in range(0, W, 40):
            for gy in range(0, H, 4):
                rect(d, gx, gy, 1, 1, (60, 120, 200))
        p = t / self.duration
        if p > 0.3:
            k = ease((p - 0.3) / 0.05)
            fy = int(-40 + k * 120)
            rect(d, 158, fy, 2, 36, (230, 230, 230))
            wv = int(math.sin(t * 6))
            rect(d, 160, fy + wv, 22, 12, (255, 205, 60))
            rect(d, 163, fy + 3 + wv, 16, 2, (200, 30, 40))
            if k >= 1.0:
                rr = int((t * 20) % 16)
                d.ellipse([159 - rr * 2, 115 - rr // 2, 159 + rr * 2, 115 + rr // 2], outline=(200, 230, 255))
        draw_trump(d, 56, 124, t)

    def overlay(self, d, t, cam):
        p = t / self.duration
        old = self.spec["old"]
        sx, sy, tw = cam.text(d, 175, 10, old, 48, (255, 255, 255), center=True)
        if p > 0.2:
            k = ease((p - 0.2) / 0.04)
            rect(d, int(sx) - 8, int(sy + 6 * cam.s), int((tw + 16) * k), 2 * cam.s, (230, 30, 40))
        if p > 0.25 and (p > 0.3 or int(t * 6) % 2 == 0):
            new = self.spec["new"]
            sx2, sy2, tw2 = cam.text(d, 175, 30, new, 64, (255, 210, 60), center=True)
            d.text((sx2 + tw2 + 10, sy2 - 10), "?!", font=font(int(64 * cam.s / 4)), fill=(255, 80, 80))


class SignboardScene(Scene):
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, H, (90, 160, 220), (190, 225, 245))
        for i, (cx, cy) in enumerate(((40, 20), (250, 14), (150, 8))):
            x = (cx + t * 6 * (i + 1)) % (W + 60) - 30
            for dx, dy, w in ((0, 4, 30), (6, 0, 18), (14, 2, 18)):
                rect(d, int(x + dx), cy + dy, w, 6, (250, 250, 255))
        items = self.spec["items"]
        for k in range(len(items)):
            y = 26 + k * 34
            show = ease((t - self.reveal(k)) / 0.3)
            if show <= 0:
                continue
            xo = int((1 - show) * 340)
            rect(d, 20 + xo, y, 280, 28, (255, 255, 255))
            rect(d, 21 + xo, y + 1, 278, 26, (20, 110, 60))
            rect(d, 24 + xo, y + 4, 34, 20, (255, 200, 60))

    def reveal(self, k):
        n = len(self.spec["items"])
        return 0.3 + k * (self.duration * 0.75) / n

    def overlay(self, d, t, cam):
        f = font(32)
        title = self.spec.get("title", "")
        tw = d.textlength(title, font=f)
        rect(d, int((FW - tw) / 2) - 16, 24, int(tw) + 32, 48, (20, 25, 50))
        d.text(((FW - tw) / 2, 32), title, font=f, fill=(255, 220, 120))
        for k, it in enumerate(self.spec["items"]):
            show = ease((t - self.reveal(k)) / 0.3)
            if show < 1:
                continue
            y = 26 + k * 34
            cam.text(d, 41, y + 8, it["year"], 24, (20, 25, 50), center=True, shadow=False)
            sx, sy, tw = cam.text(d, 66, y + 6, it["old"], 40, (255, 255, 255))
            since = t - self.reveal(k) - 0.3
            if since > 0.5:
                rect(d, int(sx) - 4, int(sy + 5 * cam.s), int((tw + 8) * ease((since - 0.5) / 0.2)),
                     cam.s * 2, (230, 40, 40))
            if since > 0.8:
                cam.text(d, 160, y + 6, "→", 40, (255, 255, 255))
                cam.text(d, 180, y + 6, it["new"], 40, (255, 220, 60))


class MeetingScene(Scene):
    PEOPLE = [{"skin": 0, "hair": 3, "top": (40, 40, 50), "suit": True},
              {"skin": 2, "hair": 4, "top": (60, 40, 40), "suit": True, "tie": (40, 80, 160)},
              {"skin": 3, "hair": 2, "top": (40, 60, 90), "long": True},
              {"skin": 1, "hair": 0, "top": (30, 30, 30), "suit": True, "tie": (40, 140, 80)}]

    def draw(self, d, t):
        vgrad(d, 0, 120, (210, 220, 235), (170, 185, 210))
        rect(d, 0, 0, W, 8, (40, 60, 110))
        # 뒤 스크린: 동해 지도
        rect(d, 104, 16, 112, 64, (20, 20, 30))
        rect(d, 107, 19, 106, 58, (40, 110, 200))
        d.polygon([(140, 19), (150, 30), (148, 50), (152, 70), (140, 77), (130, 60), (134, 30)], fill=(70, 160, 80))
        d.polygon([(213, 30), (190, 40), (178, 60), (170, 77), (213, 77)], fill=(70, 160, 80))
        # 깃발 줄
        rnd = random.Random(5)
        for fx in list(range(8, 96, 14)) + list(range(226, 316, 14)):
            rect(d, fx + 5, 30, 1, 60, (120, 120, 130))
            c1 = rnd.choice([(200, 30, 40), (40, 60, 150), (30, 140, 70), (240, 200, 40)])
            c2 = rnd.choice([(245, 245, 245), (20, 20, 20), (200, 30, 40)])
            wv = int(math.sin(t * 2 + fx) * 1)
            rect(d, fx - 4, 30 + wv, 10, 4, c1)
            rect(d, fx - 4, 34 + wv, 10, 4, c2)
        rect(d, 0, 120, W, 60, (120, 90, 70))
        talker = int(t / 2.5) % 4
        for i, x in enumerate((50, 120, 200, 270)):
            draw_person(d, x, 122 + (i % 2) * 2, t, talker == i and int(t * 9) % 2 == 0, self.PEOPLE[i],
                        gesture=False)
        rect(d, 10, 118, 300, 10, (150, 110, 80))
        rect(d, 20, 128, 280, 30, (110, 80, 60))
        for i, x in enumerate((50, 120, 200, 270)):
            rect(d, x - 6, 112, 12, 7, (245, 245, 245))
            rect(d, x - 1, 102, 2, 12, (40, 40, 40))

    def overlay(self, d, t, cam):
        cam.text(d, 160, 82, self.spec.get("sign", ""), 32, (40, 60, 110), center=True, shadow=False)
        if t > self.duration * 0.4:
            cam.text(d, 160, 40, "EAST SEA", 40, (255, 255, 255), center=True)
            cam.text(d, 160, 56, "SEA OF JAPAN?", 24, (255, 255, 255), center=True)


class ReactionScene(Scene):
    zoomable = False

    def panel_for(self, who):
        for p in self.spec["panels"]:
            if p["speaker"] == who:
                return p
        return None

    def focus_at(self, t):
        return (160, 70)

    def draw(self, d, t):
        p = self.panel_for(self.speaker(t))
        if p:
            draw_flag(d, p["flag"], 0, 0, W, H)
            rect(d, 90, 20, 140, 112, (20, 20, 30))
            rect(d, 94, 24, 132, 104, (60, 70, 90))
            draw_person(d, 160, 128, t, self.talking(t, p["speaker"]), p)
            rect(d, 90, 124, 140, 8, (20, 20, 30))
        else:
            rect(d, 0, 0, W, H, (20, 22, 40))
            for i, pp in enumerate(self.spec["panels"]):
                x = 6 + i * 78
                draw_flag(d, pp["flag"], x, 10, 74, 44)
                rect(d, x, 54, 74, 76, (60, 70, 90))
                draw_person(d, x + 37, 132, t, False, pp, gesture=False)

    def overlay(self, d, t, cam):
        rect(d, 32, 28, 200, 44, (230, 120, 30))
        d.text((48, 34), self.spec.get("label", "HYPOTHETICAL"), font=font(32), fill=(255, 255, 255))
        p = self.panel_for(self.speaker(t))
        if not p:
            for i, pp in enumerate(self.spec["panels"]):
                cam.text(d, 43 + i * 78, 108, pp["speaker"], 24, (255, 255, 255), center=True)


class ChartScene(Scene):
    zoomable = False

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (18, 22, 45))
        for y in range(30, 130, 10):
            rect(d, 0, y, W, 1, (28, 34, 64))
        bars = self.spec["bars"]
        grow = ease((t - 0.5) / 2.0)
        for i, b in enumerate(bars):
            y = 34 + i * 24
            w = int(170 * b["value"] / 100 * grow)
            rect(d, 112, y, w, 16, tuple(b["color"]))
            rect(d, 112, y + 14, w, 2, lerp(tuple(b["color"]), (0, 0, 0), 0.4))

    def overlay(self, d, t, cam):
        title = self.spec.get("title", "")
        f = font(32)
        d.text(((FW - d.textlength(title, font=f)) / 2, 40), title, font=f, fill=(255, 230, 120))
        grow = ease((t - 0.5) / 2.0)
        for i, b in enumerate(self.spec["bars"]):
            y = 34 + i * 24
            cam.text(d, 8, y + 2, b["label"], 32, (255, 255, 255))
            w = 170 * b["value"] / 100 * grow
            cam.text(d, 116 + w, y + 2, f"{round(b['value'] * grow)}%", 32, (255, 255, 255))


class InterviewScene(Scene):
    def who(self, t):
        sp = self.speaker(t)
        if sp and self.spec.get("person2") and sp == self.second_speaker():
            return "person2"
        return "person"

    def second_speaker(self):
        names = [e["line"]["speaker"] for e in self.timeline if e["line"]["speaker"] != "Reporter"]
        uniq = list(dict.fromkeys(names))
        return uniq[1] if len(uniq) > 1 else None

    def first_speaker(self):
        names = [e["line"]["speaker"] for e in self.timeline if e["line"]["speaker"] != "Reporter"]
        return names[0] if names else None

    def focus_at(self, t):
        return (160, 70)

    def draw(self, d, t):
        if self.spec.get("bg") == "studio":
            rect(d, 0, 0, W, H, (70, 50, 40))
            for sy in range(10, 130, 30):
                rect(d, 0, sy + 24, W, 4, (110, 80, 50))
                rnd = random.Random(sy)
                x = 4
                while x < W:
                    bw = rnd.randint(4, 8)
                    rect(d, x, sy + rnd.randint(4, 10), bw, 24, rnd.choice(
                        [(160, 50, 50), (50, 90, 150), (200, 170, 80), (60, 120, 70), (220, 220, 210)]))
                    x += bw + 1
        else:
            vgrad(d, 0, 100, (120, 180, 230), (200, 225, 240))
            rnd = random.Random(9)
            for bx in range(0, W, 30):
                bh = rnd.randint(40, 90)
                c = rnd.choice([(150, 150, 160), (180, 170, 150), (120, 130, 150)])
                rect(d, bx, 110 - bh, 28, bh, c)
                for wy in range(110 - bh + 5, 105, 8):
                    for wx in range(bx + 4, bx + 24, 7):
                        rect(d, wx, wy, 4, 4, (200, 225, 245))
            rect(d, 0, 110, W, 70, (150, 150, 150))
            rect(d, 0, 110, W, 3, (110, 110, 110))
            cx = int((t * 80) % (W + 120)) - 60
            rect(d, cx, 118, 40, 12, (200, 50, 50))
            rect(d, cx + 8, 112, 22, 7, (200, 50, 50))
            rect(d, cx + 10, 113, 8, 5, (180, 220, 240))
            rect(d, cx + 5, 128, 6, 5, (30, 30, 30))
            rect(d, cx + 28, 128, 6, 5, (30, 30, 30))
            for tx in (30, 290):
                rect(d, tx, 70, 4, 45, (100, 70, 40))
                d.ellipse([tx - 14, 46, tx + 18, 80], fill=(60, 150, 70))
        key = self.who(t)
        p = self.spec.get(key) or self.spec["person"]
        name = self.first_speaker() if key == "person" else self.second_speaker()
        draw_person(d, 160, 128, t, bool(name) and self.talking(t, name), p)
        # 기자 마이크
        rect(d, 196, 104, 30, 6, (240, 195, 160))
        rect(d, 186, 100, 12, 12, (230, 40, 40))
        rect(d, 180, 102, 7, 8, (40, 40, 40))

    def overlay(self, d, t, cam):
        label = self.spec.get("label", "")
        key = self.who(t)
        if self.spec.get("person2"):
            label = f"{self.first_speaker() if key == 'person' else self.second_speaker()} · {label}"
        f = font(32)
        rect(d, 32, 28, int(d.textlength(label, font=f)) + 32, 44, (40, 40, 60))
        d.text((48, 34), label, font=f, fill=(255, 255, 255))
        cam.text(d, 191, 103, "K", 16, (255, 255, 255), center=True, shadow=False)


SCENES = {"title": TitleScene, "anchor": AnchorScene, "rally": RallyScene, "map": MapScene,
          "signboard": SignboardScene, "meeting": MeetingScene, "reaction": ReactionScene,
          "chart": ChartScene, "interview": InterviewScene}


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


SPEAKER_COLOR = {"Anchor": (40, 70, 160), "Reporter": (200, 30, 40)}


def draw_subtitle(d, scene, t):
    _, e = scene.line_at(t)
    if e is None:
        return
    l = e["line"]
    text = l["text"]
    n = int(len(text) * min(1.0, (t - e["start"]) / max(0.1, e["adur"] * 0.85)))
    f = font(32)
    rows = wrap(d, text, f, FW - 160)
    box_h = 20 + len(rows) * 40
    y0 = 652 - box_h
    col = SPEAKER_COLOR.get(l["speaker"], (90, 90, 110))
    rect(d, 48, y0, FW - 96, box_h, (0, 0, 0))
    rect(d, 48, y0, 6, box_h, col)
    lw = d.textlength(l["speaker"], font=f)
    rect(d, 48, y0 - 40, int(lw) + 24, 40, col)
    d.text((60, y0 - 38), l["speaker"], font=f, fill=(255, 255, 255))
    pos = 0
    for i, row in enumerate(rows):
        start = text.find(row, pos)
        part = row[:max(0, n - start)]
        pos = start + len(row)
        d.text((76, y0 + 10 + i * 40), part, font=f, fill=(255, 255, 255))


def draw_ticker(d, ticker, gt):
    rect(d, 0, 664, FW, 56, (15, 20, 50))
    text = "   ◆   ".join(ticker) + "   ◆   "
    f = font(32)
    tw = d.textlength(text, font=f)
    x = 200 - (gt * 150) % tw
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


def bgm(total):
    """128BPM 칩튠 비트 (킥 + 하이햇 + 베이스)."""
    n = int(SR * total)
    out = np.zeros(n)
    beat = 60 / 128
    rng = np.random.default_rng(3)
    kick_t = np.arange(int(SR * 0.12)) / SR
    kick = np.sin(2 * np.pi * (110 - 400 * kick_t) * kick_t) * np.exp(-kick_t * 30)
    hat = rng.normal(0, 1, int(SR * 0.03)) * np.exp(-np.arange(int(SR * 0.03)) / SR * 150)
    bass_notes = [55, 55, 65.4, 49]
    i = 0
    while i * beat / 2 < total:
        s = int(i * beat / 2 * SR)
        if i % 2 == 0:
            out[s:s + len(kick)] += kick[: n - s] * 0.5
        else:
            out[s:s + len(hat)] += hat[: n - s] * 0.15
        note = bass_notes[(i // 8) % 4] * (2 if i % 4 == 3 else 1)
        b = square(note, beat / 2 * 0.8, 0.12)
        out[s:s + len(b)] += b[: n - s]
        i += 1
    return out


def build_audio(scenes, total):
    buf = np.zeros(int(SR * (total + 1)))

    def add(start, sig):
        i = int(start * SR)
        buf[i:i + len(sig)] += sig[: len(buf) - i]

    music = bgm(total) * 0.35
    voice = np.zeros_like(buf)
    t0 = 0.0
    for sc in scenes:
        if isinstance(sc, TitleScene):
            for k, nfreq in enumerate([523, 659, 784, 1047, 784, 1047]):
                add(t0 + 0.2 + k * 0.14, square(nfreq, 0.13, 0.18))
                add(t0 + 0.2 + k * 0.14, square(nfreq / 2, 0.13, 0.10))
        else:  # 장면 전환 휙 소리
            wt = np.arange(int(SR * 0.25)) / SR
            add(t0, np.random.default_rng(1).normal(0, 1, len(wt)) * np.sin(np.pi * wt / 0.25) * 0.08)
        if isinstance(sc, RallyScene):
            n = int(SR * sc.duration)
            noise = np.convolve(np.random.default_rng(0).normal(0, 1, n), np.ones(40) / 40, "same")
            swell = 0.5 + 0.5 * np.abs(np.sin(np.arange(n) / SR * 1.3))
            add(t0, noise * 0.25 * swell)
        for e in sc.timeline:
            i = int((t0 + e["start"]) * SR)
            a = e["audio"]
            voice[i:i + len(a)] += a[: len(voice) - i]
        t0 += sc.duration
    # 목소리 나올 때 배경음 낮추기 (더킹)
    k = SR // 10
    env = np.convolve(np.abs(voice), np.ones(k) / k, "same")
    duck = 1 - 0.6 * np.clip(env * 20, 0, 1)
    m = min(len(buf), len(music))
    buf[:m] += music[:m] * duck[:m]
    buf += voice
    buf = np.clip(buf, -1, 1)
    return (buf * 32767).astype(np.int16)


# ---------------------------------------------------------------- 렌더

def render(ep_path):
    with open(ep_path, encoding="utf-8") as fp:
        ep = json.load(fp)
    voices = Voice(ep.get("lang", "en"))
    vcfg = ep.get("voices", {})
    scenes = []
    for i, s in enumerate(ep["scenes"]):
        print(f"  TTS 장면 {i + 1}/{len(ep['scenes'])} ({s['type']})", flush=True)
        scenes.append(SCENES[s["type"]](s, voices, vcfg))
    total = sum(s.duration for s in scenes)
    print(f"  총 길이: {int(total // 60)}분 {int(total % 60)}초", flush=True)
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
         "-i", wav_path, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k", "-shortest", out],
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
        # 줄마다 줌 컷을 바꿔서 템포감
        li, _ = sc.line_at(t)
        zi = 0 if getattr(sc, "zoomable", True) is False or li < 0 else li % 2
        s, cw, ch = ZOOMS[zi]
        fx, fy = sc.focus_at(t)
        ox = max(0, min(W - cw, fx - cw // 2))
        oy = max(0, min(H - ch, fy - ch // 2))
        img = low.crop((ox, oy, ox + cw, oy + ch)).resize((FW, FH), Image.NEAREST)
        cam = Cam(s, ox, oy)
        d = ImageDraw.Draw(img)
        sc.overlay(d, t, cam)
        if not isinstance(sc, TitleScene):
            draw_subtitle(d, sc, t)
            draw_ticker(d, ep.get("ticker", []), gt)
        # 장면 전환: 짧은 화이트 플래시
        if t < 0.12 and idx > 0:
            img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), 1 - t / 0.12)
        proc.stdin.write(img.tobytes())
        if fi % (FPS * 30) == 0:
            print(f"  렌더 {gt:6.1f}s / {total:.1f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("완료:", out)
    return out


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "episodes/trump_ocean.json")
