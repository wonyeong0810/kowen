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
GAP = 0.06               # 대사 사이 쉼 (초)
EMPH = 1.0               # 강조 효과 지속 (초)
ZOOMS = [(4, 320, 180), (5, 256, 144)]   # (배율, 잘라낼 폭, 높이) - 정수 배율이라 픽셀이 고르게 유지됨

VIEW_R = 1280            # 오른쪽 끝 (숏츠에서는 잘린 화면의 오른쪽 끝)
SAFE_X = 0               # 왼쪽 위 표시 위치 보정 (숏츠에서 가로 장면을 잘라낼 때)
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
            return punch(np.load(path))
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
        return punch(x)


def punch(x):
    """긴 쉼 줄이기 + 컴프레션으로 또렷하고 빠르게."""
    hop = SR // 100
    n = len(x) // hop
    quiet = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)) < 0.012 for i in range(n)])
    keep, run = [], 0
    for i in range(n):
        run = run + 1 if quiet[i] else 0
        if run <= 7:          # 무음은 최대 70ms만 남김
            keep.append(x[i * hop:(i + 1) * hop])
    x = np.concatenate(keep + [x[n * hop:]]) if keep else x
    x = np.tanh(2.2 * x) / np.tanh(2.2)
    return (x * 0.95).astype(np.float32)


def parse_emph(text):
    """'*강조*' 표시를 떼어내고 (깨끗한 문장, [(시작, 끝, 단어)]) 반환."""
    clean, spans, i = "", [], 0
    parts = text.split("*")
    for k, part in enumerate(parts):
        if k % 2 == 1:
            spans.append((len(clean), len(clean) + len(part), part))
        clean += part
    return clean, spans


# ---------------------------------------------------------------- 캐릭터

SKINS = [(240, 195, 160), (200, 140, 100), (130, 85, 60), (250, 215, 185)]
HAIRS = [(40, 28, 20), (120, 80, 40), (230, 200, 120), (110, 110, 115), (20, 20, 20)]


def draw_person(d, cx, by, t, talking, p, gesture=True):
    """상반신 인물 (앵커/시민/전문가 공용). by = 허리선 y."""
    if p.get("char"):
        return draw_char(d, cx, by, t, talking, p["char"])
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


# ---------------------------------------------------------------- 실존 인물 캐리커처
# 특징(머리 모양, 안경, 옷차림)을 과장해서 누군지 알아보게 한다.

def _mouth(d, cx, y, talking, w=7, lips=(150, 80, 70)):
    if talking:
        rect(d, cx - w // 2, y - 1, w, 3, (120, 30, 40))
        rect(d, cx - w // 2 + 1, y - 1, w - 2, 1, (250, 250, 250))
    else:
        rect(d, cx - w // 2, y, w, 1, lips)


def _eyes(d, cx, y, t, gap=4, brow=(30, 25, 20), eye=(20, 20, 25), squint=False):
    blink = ((t + cx * 0.37) % 3.7) < 0.12
    for ex in (cx - gap - 2, cx + gap):
        rect(d, ex - 1, y - 3, 4, 1, brow)
        if blink or squint:
            rect(d, ex, y + 1, 3, 1, eye)
        else:
            rect(d, ex, y, 2, 3, eye)


def _suit(d, cx, by, top, shirt=(235, 235, 240), tie=(190, 30, 40), w=36):
    rect(d, cx - w // 2, by - 24, w, 26, top)
    rect(d, cx - w // 2 - 2, by - 18, 2, 20, top)
    rect(d, cx + w // 2, by - 18, 2, 20, top)
    if shirt:
        rect(d, cx - 5, by - 24, 10, 10, shirt)
        rect(d, cx - 3, by - 16, 6, 4, shirt)
    if tie:
        rect(d, cx - 1, by - 23, 3, 16, tie)
        rect(d, cx - 2, by - 24, 5, 2, tie)


def _glasses(d, cx, y, gap=4, c=(20, 20, 20)):
    for ex in (cx - gap - 4, cx + gap - 1):
        d.rectangle([ex, y - 2, ex + 5, y + 3], outline=c)
    rect(d, cx - 1, y, 3, 1, c)


def draw_char(d, cx, by, t, talking, key):
    skin = (240, 195, 160)
    if key == "kim_jong_un":
        # 검은 인민복(차이나 칼라), 통통한 얼굴, 옆을 바짝 민 투블럭 + 위로 솟은 앞머리
        sk = (242, 200, 165)
        _suit(d, cx, by, (25, 25, 30), shirt=None, tie=None, w=42)
        rect(d, cx - 6, by - 25, 12, 3, (40, 40, 45))           # 차이나 칼라
        for yy in range(by - 20, by, 5):
            rect(d, cx, yy, 1, 1, (90, 90, 95))                    # 단추
        rect(d, cx - 4, by - 29, 9, 6, sk)
        rect(d, cx - 11, by - 47, 23, 21, sk)                       # 넓은 얼굴
        rect(d, cx - 12, by - 38, 2, 8, sk)
        rect(d, cx + 11, by - 38, 2, 8, sk)
        rect(d, cx - 10, by - 31, 21, 4, (235, 190, 155))           # 볼살/턱
        rect(d, cx - 11, by - 47, 2, 9, (90, 85, 85))               # 민 옆머리
        rect(d, cx + 10, by - 47, 2, 9, (90, 85, 85))
        rect(d, cx - 10, by - 54, 21, 8, (15, 15, 15))              # 위로 솟은 윗머리
        rect(d, cx - 8, by - 56, 18, 3, (15, 15, 15))
        rect(d, cx - 10, by - 47, 8, 2, (15, 15, 15))
        _eyes(d, cx, by - 40, t, gap=4, brow=(15, 15, 15), squint=True)
        rect(d, cx, by - 36, 2, 2, (220, 170, 135))
        _mouth(d, cx, by - 32, talking)
    elif key == "kim_yo_jong":
        # 뒤로 묶은 검은 머리, 앞머리 옆가르마, 검은 정장
        _suit(d, cx, by, (25, 25, 30), shirt=None, tie=None)
        rect(d, cx - 5, by - 24, 10, 4, (40, 40, 45))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 46, 17, 19, skin)
        rect(d, cx - 9, by - 50, 19, 6, (15, 15, 15))
        rect(d, cx - 9, by - 46, 3, 8, (15, 15, 15))
        rect(d, cx - 6, by - 46, 9, 2, (15, 15, 15))               # 옆으로 쓸어넘긴 앞머리
        rect(d, cx + 8, by - 48, 5, 7, (15, 15, 15))               # 뒤로 묶은 머리
        _eyes(d, cx, by - 39, t, gap=3, brow=(15, 15, 15), squint=True)
        rect(d, cx, by - 35, 2, 2, (220, 170, 135))
        _mouth(d, cx, by - 31, talking, w=5, lips=(170, 70, 80))
    elif key == "lee_jae_myung":
        # 옆가르마 검은 머리(볼륨), 갸름한 얼굴, 네이비 정장 + 파란 넥타이
        _suit(d, cx, by, (25, 35, 75), tie=(40, 90, 180))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 47, 17, 20, skin)
        rect(d, cx - 6, by - 29, 13, 2, (230, 185, 150))
        rect(d, cx - 9, by - 51, 19, 6, (25, 22, 20))
        rect(d, cx - 9, by - 52, 12, 2, (25, 22, 20))
        rect(d, cx + 2, by - 47, 8, 3, (25, 22, 20))               # 오른쪽으로 넘긴 앞머리
        rect(d, cx - 9, by - 46, 2, 6, (25, 22, 20))
        rect(d, cx - 4, by - 51, 2, 1, (120, 120, 120))            # 흰머리 몇 가닥
        _eyes(d, cx, by - 39, t, gap=4, brow=(25, 22, 20))
        rect(d, cx, by - 35, 2, 3, (220, 170, 135))
        _mouth(d, cx, by - 31, talking)
    elif key == "han_dong_hoon":
        # 검은 뿔테 안경, 단정하게 넘긴 머리, 마른 얼굴, 슬림 정장 + 니트 타이
        _suit(d, cx, by, (30, 30, 35), tie=(120, 30, 40), w=32)
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 7, by - 47, 15, 20, skin)
        rect(d, cx - 8, by - 51, 17, 6, (20, 20, 20))
        rect(d, cx - 8, by - 46, 2, 5, (20, 20, 20))
        rect(d, cx - 2, by - 52, 10, 2, (20, 20, 20))              # 위로 넘긴 앞머리
        _eyes(d, cx, by - 39, t, gap=3, brow=(20, 20, 20))
        _glasses(d, cx, by - 38, gap=3)
        rect(d, cx, by - 34, 2, 2, (220, 170, 135))
        _mouth(d, cx, by - 31, talking, w=5)
    elif key == "chung_dong_young":
        # 은발, 둥근 얼굴, 회색 정장
        _suit(d, cx, by, (70, 70, 80), tie=(60, 90, 160))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 9, by - 46, 19, 19, skin)
        rect(d, cx - 10, by - 50, 21, 6, (215, 215, 220))
        rect(d, cx - 10, by - 45, 3, 7, (215, 215, 220))
        rect(d, cx + 8, by - 45, 3, 5, (215, 215, 220))
        _eyes(d, cx, by - 39, t, gap=4, brow=(170, 170, 175))
        rect(d, cx - 5, by - 33, 2, 1, (225, 175, 145))            # 미소 주름
        rect(d, cx + 4, by - 33, 2, 1, (225, 175, 145))
        rect(d, cx, by - 35, 2, 3, (220, 170, 135))
        _mouth(d, cx, by - 31, talking)
    elif key == "putin":
        # 많이 벗겨진 옅은 머리, 좁고 창백한 얼굴, 차가운 눈, 짙은 정장
        sk = (245, 215, 195)
        _suit(d, cx, by, (30, 32, 40), tie=(80, 80, 110), w=34)
        rect(d, cx - 3, by - 28, 7, 5, sk)
        rect(d, cx - 7, by - 47, 15, 20, sk)
        rect(d, cx - 8, by - 49, 17, 3, sk)                         # 넓은 이마
        rect(d, cx - 8, by - 46, 2, 6, (190, 170, 140))             # 옆머리만 남음
        rect(d, cx + 7, by - 46, 2, 6, (190, 170, 140))
        rect(d, cx - 4, by - 50, 9, 1, (200, 180, 150))             # 얇게 남은 윗머리
        _eyes(d, cx, by - 39, t, gap=3, brow=(200, 180, 150), eye=(60, 80, 110), squint=True)
        rect(d, cx, by - 35, 2, 3, (230, 190, 170))
        rect(d, cx - 5, by - 33, 1, 3, (230, 190, 170))             # 팔자 주름
        rect(d, cx + 5, by - 33, 1, 3, (230, 190, 170))
        _mouth(d, cx, by - 30, talking, w=5, lips=(190, 120, 110))
    elif key == "zelensky":
        # 올리브색 맨투맨(삼지창 문양), 짧은 갈색 머리, 수염 자국
        top = (75, 85, 55)
        rect(d, cx - 18, by - 24, 36, 26, top)
        rect(d, cx - 20, by - 18, 2, 20, top)
        rect(d, cx + 18, by - 18, 2, 20, top)
        rect(d, cx - 5, by - 25, 10, 2, (60, 70, 45))
        rect(d, cx + 6, by - 18, 3, 4, (240, 200, 60))              # 삼지창 문양
        rect(d, cx + 7, by - 20, 1, 2, (240, 200, 60))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 46, 17, 19, skin)
        rect(d, cx - 9, by - 50, 19, 5, (70, 50, 35))
        rect(d, cx - 9, by - 46, 2, 4, (70, 50, 35))
        rect(d, cx + 7, by - 46, 2, 4, (70, 50, 35))
        rect(d, cx - 8, by - 33, 17, 6, (150, 115, 90))              # 짧은 수염
        rect(d, cx - 4, by - 33, 9, 1, (150, 115, 90))
        _eyes(d, cx, by - 39, t, gap=4, brow=(70, 50, 35))
        rect(d, cx - 6, by - 36, 3, 1, (215, 170, 140))              # 다크서클
        rect(d, cx + 4, by - 36, 3, 1, (215, 170, 140))
        rect(d, cx, by - 35, 2, 3, (220, 170, 135))
        _mouth(d, cx, by - 31, talking)
    elif key == "musk":
        # 검은 티셔츠, 뒤로 넘긴 짙은 머리(살짝 M자), 넓은 턱, 활짝 웃는 입
        top = (20, 20, 22)
        rect(d, cx - 19, by - 24, 38, 26, top)
        rect(d, cx - 21, by - 18, 2, 20, top)
        rect(d, cx + 19, by - 18, 2, 20, top)
        rect(d, cx - 4, by - 24, 8, 3, skin)
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 9, by - 47, 19, 21, skin)
        rect(d, cx - 8, by - 28, 17, 2, (230, 185, 150))             # 넓은 턱
        rect(d, cx - 9, by - 51, 19, 5, (55, 40, 30))
        rect(d, cx - 9, by - 47, 2, 6, (55, 40, 30))
        rect(d, cx + 8, by - 47, 2, 6, (55, 40, 30))
        rect(d, cx - 6, by - 47, 3, 1, skin)                          # M자 이마
        rect(d, cx + 4, by - 47, 3, 1, skin)
        _eyes(d, cx, by - 40, t, gap=4, brow=(55, 40, 30))
        rect(d, cx, by - 36, 2, 3, (220, 170, 135))
        if talking:
            _mouth(d, cx, by - 31, True, w=9)
        else:
            rect(d, cx - 5, by - 32, 11, 2, (250, 250, 250))          # 활짝 웃는 이
            rect(d, cx - 5, by - 30, 11, 1, (150, 80, 70))
    elif key == "macron":
        # 뒤로 넘긴 갈색 웨이브 머리(관자놀이 흰머리), 파란 눈, 갸름한 얼굴, 네이비 정장
        _suit(d, cx, by, (25, 30, 60), tie=(30, 40, 90), w=34)
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 47, 17, 20, skin)
        rect(d, cx - 9, by - 51, 19, 6, (95, 70, 50))
        rect(d, cx - 6, by - 52, 14, 2, (95, 70, 50))
        rect(d, cx - 9, by - 46, 2, 4, (170, 160, 150))              # 관자놀이 흰머리
        rect(d, cx + 8, by - 46, 2, 4, (170, 160, 150))
        _eyes(d, cx, by - 39, t, gap=4, brow=(95, 70, 50), eye=(50, 90, 150))
        rect(d, cx, by - 35, 2, 3, (220, 170, 135))
        _mouth(d, cx, by - 31, talking, w=5)
    elif key in ("student", "student2"):
        # 후드티 고등학생 (두 가지 색)
        top = (200, 60, 60) if key == "student" else (60, 110, 180)
        rect(d, cx - 17, by - 24, 34, 26, top)
        rect(d, cx - 7, by - 26, 14, 4, lerp(top, (0, 0, 0), 0.3))   # 후드
        rect(d, cx - 2, by - 20, 1, 8, (240, 240, 240))               # 끈
        rect(d, cx + 2, by - 20, 1, 8, (240, 240, 240))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 46, 17, 19, skin)
        hc = (40, 28, 20) if key == "student" else (180, 130, 60)
        rect(d, cx - 9, by - 50, 19, 6, hc)
        rect(d, cx - 9, by - 46, 3, 7, hc)
        rect(d, cx + 7, by - 46, 3, 7, hc)
        _eyes(d, cx, by - 39, t, gap=4, brow=hc)
        rect(d, cx, by - 35, 2, 2, (220, 170, 135))
        _mouth(d, cx, by - 31, talking)
    elif key == "riot_police":
        # 진압복 + 헬멧 + 바이저
        top = (30, 35, 45)
        rect(d, cx - 19, by - 24, 38, 26, top)
        rect(d, cx - 12, by - 22, 24, 10, (45, 50, 60))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 46, 17, 19, skin)
        d.ellipse([cx - 11, by - 56, cx + 11, by - 38], fill=(25, 30, 40))
        rect(d, cx - 9, by - 44, 19, 8, (90, 110, 130))               # 바이저
        rect(d, cx - 9, by - 44, 19, 1, (150, 170, 190))
        _mouth(d, cx, by - 31, talking)
    elif key in ("mbs", "gulf"):
        # 걸프 지도자: 흰 토브 + 구트라(빨강 체크=mbs / 흰색) + 아갈(검은 줄), 검은 수염
        rect(d, cx - 18, by - 24, 36, 26, (245, 245, 240))
        rect(d, cx - 1, by - 24, 2, 10, (215, 215, 210))
        cloth = (210, 50, 55) if key == "mbs" else (245, 245, 245)
        d.polygon([(cx - 13, by - 50), (cx + 13, by - 50), (cx + 16, by - 20), (cx - 16, by - 20)], fill=cloth)
        if key == "mbs":
            for yy in range(by - 48, by - 20, 4):
                for xx in range(cx - 14, cx + 15, 4):
                    rect(d, xx, yy, 1, 1, (245, 240, 240))
        rect(d, cx - 8, by - 44, 17, 18, skin)
        rect(d, cx - 12, by - 52, 25, 4, (20, 20, 20))                    # 아갈
        rect(d, cx - 8, by - 32, 17, 6, (25, 22, 20))                     # 수염
        rect(d, cx - 4, by - 33, 9, 1, (25, 22, 20))
        _eyes(d, cx, by - 39, t, gap=4, brow=(25, 22, 20))
        rect(d, cx, by - 36, 2, 3, (220, 170, 135))
        _mouth(d, cx, by - 30, talking, lips=(120, 60, 50))
    elif key == "pak_officer":
        # 파키스탄군 장교: 카키 정복 + 녹색 베레 + 콧수염
        top = (150, 135, 95)
        rect(d, cx - 18, by - 24, 36, 26, top)
        rect(d, cx - 1, by - 24, 2, 26, (120, 105, 70))
        for sx in (-12, 9):
            rect(d, cx + sx, by - 22, 4, 3, (230, 200, 80))
        rect(d, cx - 3, by - 28, 7, 5, (200, 150, 110))
        rect(d, cx - 8, by - 46, 17, 19, (200, 150, 110))
        rect(d, cx - 11, by - 52, 22, 7, (30, 90, 50))                    # 베레
        rect(d, cx + 6, by - 50, 3, 3, (230, 200, 80))
        _eyes(d, cx, by - 39, t, gap=4)
        rect(d, cx - 4, by - 33, 9, 2, (30, 25, 20))                      # 콧수염
        _mouth(d, cx, by - 30, talking)
    elif key == "hazmat":
        # 방역복 + 고글 + 마스크
        rect(d, cx - 18, by - 24, 36, 26, (235, 235, 230))
        d.ellipse([cx - 11, by - 54, cx + 11, by - 26], fill=(235, 235, 230))
        rect(d, cx - 7, by - 44, 15, 14, skin)
        rect(d, cx - 8, by - 42, 17, 5, (90, 140, 170))                   # 고글
        rect(d, cx - 6, by - 35, 13, 6, (120, 180, 200))                  # 마스크
    elif key == "tedros":
        # WHO 사무총장: 짧은 머리, 콧수염, 정장
        sk = (120, 80, 55)
        _suit(d, cx, by, (30, 35, 50), tie=(60, 120, 190))
        rect(d, cx - 3, by - 28, 7, 5, sk)
        rect(d, cx - 8, by - 46, 17, 19, sk)
        rect(d, cx - 8, by - 49, 17, 4, (25, 22, 20))
        _eyes(d, cx, by - 39, t, gap=4, brow=(25, 22, 20))
        rect(d, cx - 4, by - 33, 9, 2, (25, 22, 20))
        _mouth(d, cx, by - 30, talking, lips=(90, 50, 40))
    elif key == "trump":
        draw_trump(d, cx, by, t, talking)
    elif key in ("general", "soldier"):
        # 육군 전투복(위장 무늬) + 베레모/전투모, 장군은 별 3개
        top = (85, 100, 60)
        rect(d, cx - 18, by - 24, 36, 26, top)
        rnd = random.Random(cx)
        for _ in range(18):
            rect(d, cx - 18 + rnd.randint(0, 32), by - 24 + rnd.randint(0, 22), 4, 3,
                 rnd.choice([(60, 70, 40), (120, 110, 70), (40, 45, 30)]))
        rect(d, cx - 3, by - 28, 7, 5, skin)
        rect(d, cx - 8, by - 46, 17, 19, skin)
        if key == "general":
            rect(d, cx - 10, by - 52, 21, 7, (60, 70, 45))          # 전투모
            rect(d, cx - 11, by - 46, 23, 2, (50, 60, 40))
            for sx in (-5, 0, 5):
                rect(d, cx + sx - 1, by - 50, 3, 3, (250, 220, 80))  # 별 3개
            for sx in (-12, 9):
                rect(d, cx + sx, by - 22, 3, 3, (250, 220, 80))
        else:
            d.ellipse([cx - 11, by - 56, cx + 11, by - 42], fill=(70, 85, 50))   # 방탄모
            rect(d, cx - 11, by - 46, 23, 2, (55, 65, 40))
        _eyes(d, cx, by - 39, t, gap=4)
        rect(d, cx, by - 35, 2, 3, (220, 170, 135))
        _mouth(d, cx, by - 31, talking)
    else:
        draw_person(d, cx, by, t, talking, {"suit": True})


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
    if kind == "jp":
        rect(d, x, y, w, h, (240, 240, 240))
        r = min(w, h) * 3 // 10
        d.ellipse([x + w // 2 - r, y + h // 2 - r, x + w // 2 + r, y + h // 2 + r], fill=(200, 30, 45))
        return
    if kind == "pl":
        rect(d, x, y, w, h // 2, (240, 240, 240))
        rect(d, x, y + h // 2, w, h - h // 2, (215, 30, 50))
        return
    if kind in ("sk", "hu"):
        cols = [(240, 240, 240), (30, 70, 160), (220, 40, 40)] if kind == "sk" else [(205, 40, 50), (240, 240, 240), (60, 130, 70)]
        for i, c in enumerate(cols):
            rect(d, x, y + i * h // 3, w, h // 3 + (h % 3 if i == 2 else 0), c)
        if kind == "sk":
            rect(d, x + w // 4, y + h // 5, w // 5, h * 3 // 5, (220, 40, 40))
            rect(d, x + w // 4 + 1, y + h // 5 + 1, w // 5 - 2, h * 3 // 5 - 2, (220, 40, 40))
            rect(d, x + w // 4 + w // 10 - 1, y + h // 5 + 2, 2, h * 3 // 5 - 4, (240, 240, 240))
        return
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
    elif kind == "us":
        draw_us_flag(d, x + 2, y, 0, w - 2, h)
    elif kind == "fr2":
        rect(d, x, y, w // 3, h, (30, 60, 160))
        rect(d, x + w // 3, y, w // 3, h, (240, 240, 240))
        rect(d, x + 2 * w // 3, y, w - 2 * w // 3, h, (220, 40, 50))
    elif kind == "sa":
        rect(d, x, y, w, h, (0, 110, 60))
        rect(d, x + w // 5, y + h // 3, 3 * w // 5, 2, (240, 240, 240))
        rect(d, x + w // 4, y + h // 2 + 4, w // 2, 2, (240, 240, 240))
        rect(d, x + 3 * w // 4 - 3, y + h // 2 + 2, 3, 6, (240, 240, 240))
    elif kind == "pk":
        rect(d, x, y, w, h, (0, 100, 50))
        rect(d, x, y, w // 4, h, (245, 245, 245))
        cx2, cy2, r = x + 5 * w // 8, y + h // 2, h // 4
        d.ellipse([cx2 - r, cy2 - r, cx2 + r, cy2 + r], fill=(245, 245, 245))
        d.ellipse([cx2 - r + r // 3, cy2 - r - r // 6, cx2 + r + r // 3, cy2 + r - r // 6], fill=(0, 100, 50))
        rect(d, cx2 + r // 2, cy2 - r // 2, 3, 3, (245, 245, 245))
    elif kind == "who":
        rect(d, x, y, w, h, (0, 120, 200))
        d.ellipse([x + w // 2 - h // 4, y + h // 4, x + w // 2 + h // 4, y + 3 * h // 4], outline=(245, 245, 245), width=2)
        rect(d, x + w // 2 - 1, y + h // 4 - 2, 2, h // 2 + 4, (245, 245, 245))
    elif kind == "kr":
        rect(d, x, y, w, h, (245, 245, 245))
        r = h // 4
        d.pieslice([x + w // 2 - r, y + h // 2 - r, x + w // 2 + r, y + h // 2 + r], 180, 360, fill=(200, 30, 50))
        d.pieslice([x + w // 2 - r, y + h // 2 - r, x + w // 2 + r, y + h // 2 + r], 0, 180, fill=(30, 60, 150))
        for tx, ty in ((x + w // 6, y + h // 5), (x + 5 * w // 6 - 6, y + h // 5),
                       (x + w // 6, y + 4 * h // 5 - 6), (x + 5 * w // 6 - 6, y + 4 * h // 5 - 6)):
            for k2 in range(3):
                rect(d, tx, ty + k2 * 2, 6, 1, (20, 20, 20))
    elif kind == "nk":
        rect(d, x, y, w, h, (30, 60, 160))
        rect(d, x, y + h // 6, w, 4 * h // 6, (245, 245, 245))
        rect(d, x, y + h // 6 + 2, w, 4 * h // 6 - 4, (210, 30, 40))
        d.ellipse([x + w // 4 - h // 5, y + h // 2 - h // 5, x + w // 4 + h // 5, y + h // 2 + h // 5], fill=(245, 245, 245))
        rect(d, x + w // 4 - 2, y + h // 2 - 2, 5, 5, (210, 30, 40))
    elif kind == "un":
        rect(d, x, y, w, h, (90, 150, 220))
        d.ellipse([x + w // 2 - h // 4, y + h // 4, x + w // 2 + h // 4, y + 3 * h // 4], outline=(245, 245, 245), width=2)
    elif kind == "ua":
        rect(d, x, y, w, h // 2, (30, 90, 200))
        rect(d, x, y + h // 2, w, h - h // 2, (250, 210, 30))
    elif kind == "ru":
        rect(d, x, y, w, h // 3, (240, 240, 240))
        rect(d, x, y + h // 3, w, h // 3, (30, 60, 170))
        rect(d, x, y + 2 * h // 3, w, h - 2 * h // 3, (210, 30, 40))
    elif kind == "eu":
        rect(d, x, y, w, h, (30, 50, 150))
        for k in range(12):
            a = k / 12 * 6.283
            rect(d, int(x + w / 2 + math.cos(a) * h / 3), int(y + h / 2 + math.sin(a) * h / 3), 2, 2, (250, 210, 30))
    elif kind == "mx":
        rect(d, x, y, w // 3, h, (20, 110, 60))
        rect(d, x + w // 3, y, w // 3, h, (240, 240, 240))
        rect(d, x + 2 * w // 3, y, w - 2 * w // 3, h, (200, 30, 40))
        d.ellipse([x + w // 2 - 6, y + h // 2 - 6, x + w // 2 + 6, y + h // 2 + 6], fill=(140, 90, 40))
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
            clean, spans = parse_emph(l["text"])
            audio = voices.say(clean, vc["sid"], vc.get("speed", 1.2))
            adur = len(audio) / SR
            emph = [(acc + adur * a / max(1, len(clean)), a, b, w) for a, b, w in spans]
            hop = SR // FPS
            env = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2)) for i in range(0, len(audio), hop)])
            self.timeline.append({"start": acc, "end": acc + adur + GAP, "adur": adur,
                                  "line": l, "audio": audio, "env": env,
                                  "clean": clean, "emph": emph})
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
            sub = self.spec.get("subtitle", "")
            d.text(((FW - d.textlength(sub, font=font(32))) / 2, y + 120), sub,
                   font=font(32), fill=(230, 230, 255))


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
        rect(d, SAFE_X + 32, 28, 96, 40, (210, 30, 40))
        if int(t * 2) % 2 == 0:
            d.ellipse([SAFE_X + 42, 40, SAFE_X + 58, 56], fill=(255, 255, 255))
        d.text((SAFE_X + 64, 32), "LIVE", font=font(32), fill=(255, 255, 255))
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
        tag = self.spec.get("tag", "LIVE")
        f2 = font(32)
        tw = int(d.textlength(tag, font=f2))
        rect(d, 1240 - tw - 32, 108, tw + 32, 40, (0, 0, 0))
        d.text((1240 - tw - 16, 112), tag, font=f2, fill=(255, 230, 120))


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
            y = 26 + k * self.row_h()
            show = ease((t - self.reveal(k)) / 0.3)
            if show <= 0:
                continue
            xo = int((1 - show) * 340)
            bh = self.row_h() - 6
            rect(d, 20 + xo, y, 280, bh, (255, 255, 255))
            rect(d, 21 + xo, y + 1, 278, bh - 2, (20, 110, 60))
            rect(d, 24 + xo, y + 3, 34, bh - 6, (255, 200, 60))

    def row_h(self):
        # 자막 상자(저해상도 y≈128부터)에 가리지 않게 항목 수에 따라 간격을 줄인다
        return min(34, 100 // max(1, len(self.spec["items"])))

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
            y = 26 + k * self.row_h()
            cam.text(d, 41, y + 8, it["year"], 24, (20, 25, 50), center=True, shadow=False)
            if "text" in it:
                cam.text(d, 66, y + 6, it["text"], 32, (255, 255, 255))
                continue
            sx, sy, tw = cam.text(d, 66, y + 6, it["old"], 40, (255, 255, 255))
            since = t - self.reveal(k) - 0.3
            if since > 0.5 and it["old"] != it["new"]:
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
        if self.spec.get("screen", True):
            rect(d, 104, 16, 112, 64, (20, 20, 30))
            rect(d, 107, 19, 106, 58, (40, 110, 200))
            d.polygon([(140, 19), (150, 30), (148, 50), (152, 70), (140, 77), (130, 60), (134, 30)],
                      fill=(70, 160, 80))
            d.polygon([(213, 30), (190, 40), (178, 60), (170, 77), (213, 77)], fill=(70, 160, 80))
        else:  # 파란 커튼 + 원형 문장
            for cx in range(96, 226, 8):
                rect(d, cx, 8, 6, 90, (30, 50, 120))
                rect(d, cx + 6, 8, 2, 90, (20, 35, 90))
            d.ellipse([140, 26, 180, 66], fill=(230, 200, 110))
            d.ellipse([146, 32, 174, 60], fill=(40, 60, 130))
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
        if not self.spec.get("screen", True):
            tag = self.spec.get("sign", "")
            tw = d.textlength(tag, font=font(32))
            rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (30, 40, 90))
            d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))
        else:
            cam.text(d, 160, 82, self.spec.get("sign", ""), 32, (40, 60, 110), center=True, shadow=False)
        if self.spec.get("screen", True) and t > self.duration * 0.4:
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
        rect(d, SAFE_X + 32, 28, 200, 44, (230, 120, 30))
        d.text((SAFE_X + 48, 34), self.spec.get("label", "HYPOTHETICAL"), font=font(32), fill=(255, 255, 255))
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
            w = int(170 * b["value"] / self.spec.get("max", 100) * grow)
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
            w = 170 * b["value"] / self.spec.get("max", 100) * grow
            v = round(b["value"] * grow)
            cam.text(d, 116 + w, y + 2, f"{v:,}{self.spec.get('unit', '%')}", 32, (255, 255, 255))


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
        rect(d, SAFE_X + 32, 28, int(d.textlength(label, font=f)) + 32, 44, (40, 40, 60))
        d.text((SAFE_X + 48, 34), label, font=f, fill=(255, 255, 255))
        cam.text(d, 191, 103, "K", 16, (255, 255, 255), center=True, shadow=False)


class QuoteScene(Scene):
    """실제 발언 인용 카드: 국기 배경 + 인물 + 이름표 + 인용문 (기자가 읽음)."""
    zoomable = False
    focus = (120, 90)   # 숏츠로 잘라도 왼쪽 인물 상자가 잘리지 않게

    def draw(self, d, t):
        draw_flag(d, self.spec.get("flag", "ca"), 0, 0, W, H)
        if self.spec.get("quote"):
            rect(d, 0, 0, W, H // 3, (0, 0, 0))
        rect(d, 20, 64, 90, 70, (20, 20, 30))
        rect(d, 23, 67, 84, 64, (60, 70, 90))
        draw_person(d, 65, 133, t, False, self.spec.get("person", {"suit": True}), gesture=False)

    def overlay(self, d, t, cam):
        f = font(32)
        q = self.spec["quote"] if self.spec.get("plain") else "\u201c" + self.spec["quote"] + "\u201d"
        rows = wrap(d, q, f, FW - 120)
        k = ease(t / 0.4)
        for i, row in enumerate(rows[:4]):
            d.text((60 - (1 - k) * 80, 24 + i * 40), row, font=f, fill=(255, 230, 120))
        name, role = self.spec.get("name", ""), self.spec.get("role", "")
        nf = 48   # 이름이 화면(숏츠는 잘린 폭) 밖으로 나가지 않게 글자 크기 줄이기
        while nf > 24 and 480 + d.textlength(name, font=font(nf)) > VIEW_R - 30:
            nf -= 4
        rf = 32
        while rf > 16 and 480 + d.textlength(role, font=font(rf)) > VIEW_R - 30:
            rf -= 4
        f = font(rf)
        rect(d, 460, 290, int(max(d.textlength(name, font=font(nf)), d.textlength(role, font=f))) + 40, 116, (0, 0, 0))
        d.text((480, 300), name, font=font(nf), fill=(255, 255, 255))
        d.text((480, 356), role, font=f, fill=(200, 200, 220))


PUTIN = {"char": "putin"}


class CallScene(Scene):
    """정상 통화: 왼쪽 트럼프, 오른쪽 푸틴 분할 화면."""
    zoomable = False

    def draw(self, d, t):
        draw_us_flag(d, 0, 0, 0, 160, 120)
        draw_flag(d, "ru", 160, 0, 160, 120)
        rect(d, 156, 0, 8, H, (10, 10, 15))
        rect(d, 10, 20, 140, 100, (60, 40, 30))
        rect(d, 170, 20, 140, 100, (90, 90, 100))
        draw_trump(d, 80, 112, t, talking=int(t / 1.6) % 2 == 0)
        draw_person(d, 240, 112, t, int(t / 1.6) % 2 == 1 and int(t * 9) % 2 == 0, PUTIN, gesture=False)
        rect(d, 225, 70, 5, 14, (30, 30, 30))   # 수화기
        rect(d, 10, 108, 140, 12, (90, 60, 40))
        rect(d, 170, 108, 140, 12, (60, 60, 70))
        # 전화선 파동
        for i in range(6):
            yy = 60 + int(math.sin(t * 8 + i) * 4)
            rect(d, 152 + (i % 2) * 8, yy + i * 3, 3, 2, (255, 220, 60))

    def overlay(self, d, t, cam):
        f = font(32)
        for x, name in ((max(40, SAFE_X), "TRUMP"), (680, "PUTIN")):
            rect(d, x, 500 - 60, int(d.textlength(name, font=f)) + 24, 40, (0, 0, 0))
            d.text((x + 12, 444), name, font=f, fill=(255, 255, 255))
        tag = self.spec.get("tag", "PHONE CALL")
        tw = d.textlength(tag, font=f)
        rect(d, int((FW - tw) / 2) - 16, 24, int(tw) + 32, 44, (200, 30, 40))
        d.text(((FW - tw) / 2, 28), tag, font=f, fill=(255, 255, 255))


class PumpScene(Scene):
    """주유소 가격판: 숫자가 올라가며 기록 경신."""
    zoomable = False
    focus = (200, 80)

    def draw(self, d, t):
        vgrad(d, 0, 120, (250, 160, 90), (120, 70, 120))
        rect(d, 0, 120, W, 60, (70, 70, 75))
        rect(d, 20, 40, 200, 10, (220, 40, 40))
        for px in (40, 190):
            rect(d, px, 50, 8, 70, (200, 200, 205))
        for i, px in enumerate((70, 140)):
            rect(d, px, 80, 28, 42, (220, 220, 225))
            rect(d, px + 4, 86, 20, 10, (20, 20, 30))
            rect(d, px + 28, 90, 6, 3, (30, 30, 30))
        rect(d, 238, 20, 70, 100, (30, 30, 40))
        rect(d, 270, 120, 6, 20, (60, 60, 60))
        cx = int((t * 40) % (W + 140)) - 140   # 트럭
        rect(d, cx, 128, 80, 30, (230, 230, 235))
        rect(d, cx + 80, 136, 26, 22, (40, 90, 170))
        rect(d, cx + 88, 140, 12, 8, (170, 210, 240))
        for wx in (cx + 10, cx + 60, cx + 92):
            rect(d, wx, 156, 10, 8, (20, 20, 20))

    def overlay(self, d, t, cam):
        rows = self.spec.get("prices", [])
        for i, r in enumerate(rows):
            k = ease((t - 0.3 - i * 0.4) / 0.8)
            val = r["value"] * k
            y = 96 + i * 130
            d.text((962, y), r["label"], font=font(32), fill=(255, 220, 120))
            d.text((962, y + 40), f"${val:0.2f}", font=font(64), fill=(255, 80, 60) if r.get("hot") else (255, 255, 255))
        title = self.spec.get("title", "")
        f = font(32)
        tw = d.textlength(title, font=f)
        rect(d, 80, 230, int(tw) + 32, 48, (20, 25, 50))
        d.text((96, 236), title, font=f, fill=(255, 255, 255))


class DMZScene(Scene):
    """비무장지대: 철책, 지뢰 경고판, 수색 중인 장병, (옵션) 폭발 연출."""
    focus = (160, 90)

    def draw(self, d, t):
        vgrad(d, 0, 70, (150, 170, 190), (200, 205, 200))
        for i, (hx, hh) in enumerate(((0, 30), (70, 40), (160, 28), (240, 44))):
            d.polygon([(hx - 40, 70), (hx + 40, 70 - hh), (hx + 120, 70)], fill=(80, 100, 70))
        rect(d, 0, 70, W, 110, (95, 115, 65))
        rnd = random.Random(4)
        for _ in range(140):
            rect(d, rnd.randint(0, W), rnd.randint(70, 180), 2, 3, rnd.choice([(70, 90, 50), (120, 130, 70)]))
        # 철책
        rect(d, 0, 76, W, 1, (150, 150, 150))
        for fx in range(0, W, 16):
            rect(d, fx, 64, 2, 22, (110, 110, 110))
        for yy in (68, 72, 80):
            for fx in range(0, W, 4):
                rect(d, fx, yy + (fx // 4) % 2, 2, 1, (170, 170, 170))
        # 지뢰 경고판
        for mx in (40, 200, 290):
            rect(d, mx, 96, 2, 12, (90, 70, 50))
            d.polygon([(mx - 7, 98), (mx + 9, 98), (mx + 1, 86)], fill=(220, 30, 30))
            rect(d, mx, 92, 2, 3, (255, 255, 255))
        # 수색 중인 장병 3명 (탐지기 휘두르기)
        boom = self.spec.get("boom")
        for i, sx in enumerate((110, 150, 190)):
            bob = int(math.sin(t * 4 + i) * 1)
            x = sx + int(math.sin(t * 0.6) * 4)
            draw_char(d, x, 148 + bob, t, False, "soldier")
            sw = int(math.sin(t * 3 + i) * 6)
            d.line([(x + 18, 136), (x + 26 + sw, 160)], fill=(40, 40, 40), width=2)
            rect(d, x + 22 + sw, 158, 10, 3, (60, 60, 60))
        if boom is not None and boom <= t < boom + 1.6:
            k = (t - boom) / 1.6
            r = int(10 + 60 * k)
            c = lerp((255, 240, 160), (120, 110, 100), k)
            d.ellipse([230 - r, 140 - r // 2, 230 + r, 140 + r // 2], fill=c)
            for j in range(10):
                a = j / 10 * 6.283
                rect(d, int(230 + math.cos(a) * r * 1.2), int(140 + math.sin(a) * r * 0.5), 3, 3, (60, 50, 40))

    def overlay(self, d, t, cam):
        tag = self.spec.get("tag", "DMZ")
        f = font(32)
        tw = d.textlength(tag, font=f)
        rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (40, 60, 40))
        d.text((SAFE_X + 48, 34), tag, font=f, fill=(255, 255, 255))
        rect(d, VIEW_R - 268, 28, 236, 40, (0, 0, 0))
        d.text((VIEW_R - 256, 32), "ILLUSTRATION", font=f, fill=(255, 230, 120))
        boom = self.spec.get("boom")
        if boom is not None and boom <= t < boom + 0.15:
            rect(d, 0, 0, FW, FH, (255, 255, 230))


class KoreaMapScene(Scene):
    """한반도 지도 + DMZ 선 + 사고 지점 표시."""
    zoomable = False
    KOREA = [(150, 6), (170, 4), (196, 14), (206, 30), (190, 44), (180, 58), (172, 70),
             (182, 84), (196, 110), (200, 140), (190, 168), (160, 176), (136, 170), (128, 150),
             (136, 120), (126, 100), (132, 80), (124, 66), (134, 46), (140, 26)]

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (30, 80, 150))
        for y in range(6, H, 12):
            for x in range(0, W, 16):
                rect(d, (x + int(t * 8) + y) % W, y, 5, 1, (60, 120, 190))
        d.polygon([(0, 0), (150, 0), (140, 26), (110, 30), (60, 20), (0, 30)], fill=(150, 140, 110))
        d.polygon(self.KOREA, fill=(90, 150, 80))
        d.polygon([(124, 66), (134, 46), (140, 26), (150, 6), (170, 4), (196, 14), (206, 30), (190, 44),
                   (180, 58), (172, 70)], fill=(120, 120, 120))       # 북한 (회색)
        d.line([(126, 72), (150, 70), (176, 66)], fill=(230, 40, 40), width=3)   # DMZ
        if int(t * 3) % 2 == 0:
            d.ellipse([139, 66, 147, 74], outline=(255, 230, 60), width=2)
        rect(d, 142, 69, 2, 2, (255, 230, 60))

    def overlay(self, d, t, cam):
        cam.text(d, 186, 22, "NORTH KOREA", 32, (255, 255, 255))
        cam.text(d, 196, 120, "SOUTH KOREA", 32, (255, 255, 255))
        cam.text(d, 196, 64, "DMZ", 40, (255, 90, 90))
        place = self.spec.get("place", "Paju–Yeoncheon")
        if place:
            cam.text(d, 40, 82, place, 32, (255, 230, 60))
        for i, row in enumerate(self.spec.get("facts", [])):
            k = ease((t - 0.5 - i * 0.6) / 0.4)
            if k > 0:
                y = 40 if self.spec.get("facts_top") else 420
                rect(d, int(40 - (1 - k) * 400), y + i * 52, 480, 44, (0, 0, 0))
                d.text((56 - (1 - k) * 400, y + 6 + i * 52), row, font=font(32), fill=(255, 255, 255))


class UNScene(Scene):
    """유엔 총회장: 연단의 발언자 + 초록 대리석 벽 + 유엔 엠블럼."""
    focus = (160, 70)

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (40, 90, 70))
        for x in range(0, W, 20):
            rect(d, x, 0, 1, H, (55, 110, 85))
        rect(d, 110, 0, 100, 120, (200, 170, 90))
        d.ellipse([130, 10, 190, 70], fill=(90, 150, 220))
        d.ellipse([140, 20, 180, 60], outline=(245, 245, 245), width=2)
        rect(d, 159, 22, 2, 36, (245, 245, 245))
        rect(d, 142, 39, 36, 2, (245, 245, 245))
        p = self.spec.get("speaker", {"suit": True})
        draw_person(d, 160, 128, t, int(t * 9) % 2 == 0, p, gesture=False)
        rect(d, 132, 118, 56, 62, (60, 40, 30))
        rect(d, 128, 114, 64, 6, (90, 60, 40))
        rect(d, 158, 104, 2, 12, (40, 40, 40))
        rect(d, 0, 150, W, 30, (30, 30, 40))
        rnd = random.Random(11)
        for x in range(4, W, 14):
            rect(d, x, 146 + rnd.randint(0, 4), 10, 10, rnd.choice(SKINS))
            rect(d, x - 1, 154, 12, 26, rnd.choice([(30, 30, 40), (50, 50, 70), (70, 60, 50)]))

    def overlay(self, d, t, cam):
        f = font(32)
        for i, txt in enumerate((self.spec.get("name", ""), self.spec.get("role", ""))):
            if txt:
                ff = font(40) if i == 0 else f
                tw = d.textlength(txt, font=ff)
                rect(d, SAFE_X + 32, 28 + i * 52, int(tw) + 32, 48, (0, 0, 0) if i else (40, 90, 160))
                d.text((SAFE_X + 48, 32 + i * 52), txt, font=ff, fill=(255, 255, 255))


class AuditScene(Scene):
    """국회 국정감사장: 왼쪽 질의 의원, 오른쪽 답변 장관."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 110, (150, 120, 90), (120, 95, 70))
        for x in range(0, W, 24):
            rect(d, x, 0, 2, 110, (110, 85, 60))
        rect(d, 100, 10, 120, 24, (30, 50, 110))
        rect(d, 0, 110, W, 70, (90, 60, 40))
        q, a = self.spec["questioner"], self.spec["answerer"]
        who = self.active(t)
        flap = int(t * 9) % 2 == 0
        draw_person(d, 80, 124, t, who == "q" and flap, q["person"], gesture=False)
        draw_person(d, 240, 124, t, who == "a" and flap, a["person"], gesture=False)
        for x in (80, 240):
            rect(d, x - 40, 132, 80, 48, (70, 45, 30))
            rect(d, x - 2, 118, 2, 16, (40, 40, 40))
            rect(d, x - 20, 136, 40, 10, (245, 245, 245))

    def active(self, t):
        _, e = self.line_at(t)
        if e is None:
            return None
        return e["line"].get("who")

    def overlay(self, d, t, cam):
        cam.text(d, 160, 14, self.spec.get("sign", "NATIONAL ASSEMBLY AUDIT"), 32, (255, 255, 255), center=True,
                 shadow=False)
        f = font(32)
        for x, side in ((80, "questioner"), (240, "answerer")):
            info = self.spec[side]
            sx, _ = cam.pt(x, 150)
            for i, txt in enumerate((info["name"], info["role"])):
                ff = font(36) if i == 0 else font(24)
                tw = d.textlength(txt, font=ff)
                rect(d, int(sx - tw / 2) - 12, 150 + i * 44, int(tw) + 24, 40, (0, 0, 0))
                d.text((sx - tw / 2, 154 + i * 44), txt, font=ff, fill=(255, 255, 255) if i == 0 else (200, 200, 220))


class XPostScene(Scene):
    """X(트위터) 게시물 카드: 프로필 캐릭터 + 계정 + 글."""
    zoomable = False

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (10, 12, 16))
        for y in range(0, H, 6):
            rect(d, 0, y, W, 1, (16, 18, 24))
        rect(d, 30, 10, 260, 116, (0, 0, 0))
        d.rectangle([30, 10, 289, 125], outline=(60, 65, 75))
        av = Image.new("RGB", (40, 40), (40, 50, 70))
        draw_person(ImageDraw.Draw(av), 20, 60, t, False, self.spec.get("person", {"suit": True}), gesture=False)
        mask = Image.new("L", (40, 40), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, 39, 39], fill=255)
        d._image.paste(av, (36, 14), mask)

    def overlay(self, d, t, cam):
        name, handle = self.spec.get("name", ""), self.spec.get("handle", "")
        d.text((320, 70), name, font=font(40), fill=(255, 255, 255))
        d.text((320 + d.textlength(name, font=font(40)) + 16, 78), handle, font=font(32), fill=(130, 140, 155))
        logo = self.spec.get("logo", "X")
        d.text((1150 - d.textlength(logo, font=font(48 if len(logo) > 1 else 64)), 64 if len(logo) > 1 else 60),
               logo, font=font(48 if len(logo) > 1 else 64), fill=(255, 255, 255))
        k = ease((t - 0.3) / 0.6)
        if k > 0:
            for i, row in enumerate(wrap(d, self.spec.get("post", ""), font(48), 920)[:4]):
                d.text((150, 230 + i * 60), row, font=font(48), fill=(255, 255, 255))
        date = self.spec.get("date", "")
        if date:
            d.text((150, 470), date, font=font(32), fill=(130, 140, 155))


class PressScene(Scene):
    """백악관 기자들 앞에서 말하는 트럼프."""
    focus = (160, 70)

    def draw(self, d, t):
        for cx in range(0, W, 10):
            rect(d, cx, 0, 8, 130, (30, 50, 120))
            rect(d, cx + 8, 0, 2, 130, (20, 35, 90))
        d.ellipse([228, 12, 268, 52], fill=(230, 200, 110))
        d.ellipse([233, 17, 263, 47], fill=(40, 60, 130))
        draw_trump(d, 160, 118, t)
        rect(d, 0, 118, W, 62, (60, 45, 35))
        # 기자들 머리 + 마이크 + 카메라
        rnd = random.Random(21)
        for x in range(-4, W, 22):
            rect(d, x, 140 + rnd.randint(0, 6), 14, 14, rnd.choice(SKINS))
            rect(d, x - 1, 136 + rnd.randint(0, 4), 16, 5, rnd.choice(HAIRS))
            rect(d, x - 3, 154, 20, 26, rnd.choice([(40, 40, 50), (70, 60, 50), (90, 90, 100)]))
        for i, mx in enumerate((120, 150, 175, 200)):
            sway = int(math.sin(t * 2 + i) * 2)
            d.line([(mx, 180), (mx + 10 + sway, 112)], fill=(30, 30, 30), width=2)
            rect(d, mx + 8 + sway, 106, 6, 8, (40, 40, 40))
        for cx in (30, 270):
            rect(d, cx, 100, 24, 16, (30, 30, 30))
            rect(d, cx + 24, 104, 8, 8, (60, 60, 60))
            if int(t * 1.5 + cx) % 5 == 0:
                rect(d, cx + 2, 96, 10, 4, (255, 255, 230))

    def overlay(self, d, t, cam):
        tag = self.spec.get("tag", "WHITE HOUSE")
        f = font(32)
        tw = d.textlength(tag, font=f)
        rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (30, 40, 90))
        d.text((SAFE_X + 48, 34), tag, font=f, fill=(255, 255, 255))


class ProtestScene(Scene):
    """학생 시위: 리세(고등학교) 건물, 피켓 든 학생 무리, (옵션) 최루가스와 진압 경찰."""
    focus = (160, 90)

    def draw(self, d, t):
        vgrad(d, 0, 60, (150, 175, 210), (205, 215, 225))
        # 리세 건물 (위쪽으로 올려 학생들이 자막에 가리지 않게)
        rect(d, 40, 8, 240, 56, (215, 200, 170))
        rect(d, 40, 4, 240, 6, (150, 90, 70))
        for wx in range(52, 270, 22):
            for wy in (16, 38):
                rect(d, wx, wy, 12, 14, (90, 120, 150))
        rect(d, 140, 42, 40, 22, (110, 70, 50))
        rect(d, 0, 64, W, 116, (120, 120, 125))
        # 피켓 든 학생 무리
        rnd = random.Random(31)
        cheer = self.spec.get("cheer", 1.0)
        for row, y in enumerate((96, 108, 120)):
            x = -6 + row * 6
            while x < W + 10:
                top = rnd.choice([(200, 60, 60), (60, 110, 180), (50, 50, 55), (220, 200, 60), (90, 160, 90)])
                ph = rnd.random() * 6.28
                bob = int(abs(math.sin(t * 5 * cheer + ph)) * 3 * cheer)
                yy = y - bob
                rect(d, x, yy + 9, 12, 22, top)
                rect(d, x + 2, yy, 8, 9, rnd.choice(SKINS))
                rect(d, x + 1, yy - 1, 10, 3, rnd.choice(HAIRS))
                if rnd.random() < 0.18:
                    rect(d, x + 5, yy - 20, 1, 18, (120, 90, 60))
                    rect(d, x - 4, yy - 30, 20, 11, (245, 245, 240))
                    rect(d, x - 2, yy - 27, 16, 2, rnd.choice([(30, 30, 30), (200, 30, 40)]))
                    rect(d, x - 2, yy - 23, 11, 2, (30, 30, 30))
                x += 13 + rnd.randint(0, 3)
        # 큰 현수막
        rect(d, 70, 68, 180, 18, (245, 245, 240))
        rect(d, 72, 70, 176, 14, (200, 30, 40))
        if self.spec.get("police"):
            for i, px in enumerate((24, 52, 268, 296)):
                draw_char(d, px, 126, t, False, "riot_police")
                rect(d, px - 14, 98, 6, 30, (50, 60, 80))                   # 방패
            for k2 in range(4):                                             # 최루가스
                gx = (int(t * 25) + k2 * 80) % (W + 60) - 30
                r = 14 + int(math.sin(t * 2 + k2) * 4)
                d.ellipse([gx - r, 70 - r // 2, gx + r, 70 + r // 2], fill=(215, 215, 210))

    def overlay(self, d, t, cam):
        cam.text(d, 160, 71, self.spec.get("banner", ""), 32, (255, 255, 255), center=True, shadow=False)
        tag = self.spec.get("tag", "")
        if tag:
            tw = d.textlength(tag, font=font(32))
            rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (30, 40, 90))
            d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))


class ClassroomScene(Scene):
    """낡은 교실: 꽉 찬 책상, 금 간 천장과 물 샘, 온도계, 쥐, 빈 교사 자리."""
    zoomable = False

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (190, 185, 160))
        rect(d, 0, 0, W, 10, (160, 155, 130))
        d.line([(60, 0), (75, 10), (70, 18), (90, 24)], fill=(110, 100, 80), width=2)   # 천장 균열
        for k2 in range(3):                                                           # 물방울
            yy = (int(t * 40) + k2 * 30) % 70 + 12
            rect(d, 88, yy, 2, 3, (90, 140, 200))
        rect(d, 84, 80, 12, 2, (90, 140, 200))
        rect(d, 110, 14, 100, 46, (40, 70, 50))                                        # 칠판
        rect(d, 108, 12, 104, 2, (120, 90, 60))
        rect(d, 140, 62, 40, 14, (120, 90, 60))                                        # 빈 교탁
        rect(d, 250, 18, 6, 40, (240, 240, 240))                                       # 온도계
        h = 30 + int(math.sin(t) * 2)
        rect(d, 251, 56 - h, 4, h, (220, 40, 40))
        d.ellipse([248, 54, 258, 64], fill=(220, 40, 40))
        rnd = random.Random(7)
        for row in range(4):                                                           # 빽빽한 학생들
            for col in range(9):
                x, y = 16 + col * 34, 92 + row * 22
                rect(d, x - 2, y + 10, 28, 4, (150, 110, 70))
                rect(d, x + 7, y, 10, 10, rnd.choice(SKINS))
                rect(d, x + 6, y - 1, 12, 3, rnd.choice(HAIRS))
                rect(d, x + 5, y + 6, 14, 6, rnd.choice([(200, 60, 60), (60, 110, 180), (90, 160, 90)]))
        rx = int((t * 30) % (W + 40)) - 20                                             # 쥐
        rect(d, rx, 172, 8, 4, (90, 80, 75))
        rect(d, rx + 8, 173, 3, 2, (90, 80, 75))
        rect(d, rx - 5, 174, 5, 1, (150, 120, 110))

    def overlay(self, d, t, cam):
        cam.text(d, 160, 26, self.spec.get("board", ""), 32, (240, 240, 230), center=True, shadow=False)
        cam.text(d, 253, 66, self.spec.get("temp", "40°C"), 24, (220, 40, 40), center=True)
        for i, row in enumerate(self.spec.get("facts", [])):
            k2 = ease((t - 0.4 - i * self.duration / max(1, len(self.spec["facts"])) * 0.8) / 0.4)
            if k2 > 0:
                tw = d.textlength(row, font=font(32))
                rect(d, int(FW - 40 - tw - 32 + (1 - k2) * 600), 100 + i * 52, int(tw) + 32, 44, (0, 0, 0))
                d.text((FW - 40 - tw - 16 + (1 - k2) * 600, 106 + i * 52), row, font=font(32), fill=(255, 230, 120))


class AirDefenseScene(Scene):
    """방공 포대: 날아오는 미사일을 요격하고, 남은 요격미사일 게이지가 줄어든다."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 120, (15, 20, 45), (60, 50, 70))
        rnd = random.Random(2)
        for _ in range(40):
            rect(d, rnd.randint(0, W), rnd.randint(0, 80), 1, 1, (200, 200, 220))
        rect(d, 0, 120, W, 60, (150, 130, 95))
        for bx in range(180, W, 26):                                    # 도시 실루엣
            bh = rnd.randint(14, 40)
            rect(d, bx, 120 - bh, 22, bh, (35, 35, 50))
        for i, lx in enumerate((40, 90)):                                # 발사대
            rect(d, lx - 14, 112, 36, 10, (90, 95, 70))
            d.polygon([(lx - 10, 112), (lx + 14, 92), (lx + 20, 96), (lx - 4, 114)], fill=(110, 115, 85))
            for wx in (lx - 10, lx + 12):
                rect(d, wx, 120, 6, 4, (30, 30, 30))
        period = 2.4
        for k2 in range(2):
            ph = (t + k2 * 1.2) % period / period
            mx, my = int(330 - ph * 170), int(10 + ph * 50)              # 날아오는 미사일
            if ph < 0.75:
                rect(d, mx, my, 6, 2, (230, 230, 230))
                rect(d, mx + 6, my, 6, 2, (255, 160, 60))
                ix, iy = int(60 + ph / 0.75 * (mx - 60)), int(100 - ph / 0.75 * (100 - my))
                rect(d, ix, iy, 2, 4, (240, 240, 200))                   # 요격미사일
                d.line([(60, 100), (ix, iy + 4)], fill=(200, 200, 190))
            elif ph < 0.95:
                r = int((ph - 0.75) / 0.2 * 14) + 3
                d.ellipse([207 - r, 47 - r, 207 + r, 47 + r], fill=(255, 200, 90))

    def overlay(self, d, t, cam):
        tag = self.spec.get("tag", "")
        tw = d.textlength(tag, font=font(32))
        rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (30, 40, 90))
        d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))
        level = 1.0 - 0.75 * min(1.0, t / max(1.0, self.duration * 0.8))  # 요격미사일 게이지
        x0, y0 = VIEW_R - 360, 100
        d.text((x0, y0 - 40), "INTERCEPTORS", font=font(32), fill=(255, 255, 255))
        rect(d, x0, y0, 300, 36, (40, 40, 50))
        col = (80, 200, 90) if level > 0.5 else (240, 180, 50) if level > 0.3 else (230, 50, 50)
        rect(d, x0 + 4, y0 + 4, int(292 * level), 28, col)
        if level <= 0.3 and int(t * 3) % 2 == 0:
            d.text((x0, y0 + 44), "RUNNING LOW", font=font(32), fill=(255, 80, 80))


class AirportScene(Scene):
    """공격받은 공항 터미널: 불길과 연기, 멈춘 비행기."""
    focus = (160, 80)

    def draw(self, d, t):
        vgrad(d, 0, 100, (90, 110, 140), (170, 160, 150))
        rect(d, 30, 50, 260, 56, (200, 200, 205))
        rect(d, 30, 46, 260, 6, (120, 125, 135))
        for wx in range(40, 285, 12):
            rect(d, wx, 60, 8, 30, (70, 110, 150))
        d.polygon([(150, 46), (190, 46), (200, 70), (170, 80), (140, 66)], fill=(40, 35, 35))   # 파손 부위
        for k2 in range(6):                                            # 연기와 불
            ph = (t * 0.6 + k2 / 6) % 1
            sx, sy = 170 + int(math.sin(k2 + t) * 8), int(60 - ph * 60)
            r = 6 + int(ph * 14)
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=lerp((70, 65, 65), (150, 150, 150), ph))
        for k2 in range(4):
            fx = 158 + k2 * 8
            fh = 6 + int(abs(math.sin(t * 8 + k2)) * 6)
            rect(d, fx, 72 - fh, 5, fh, (255, 150 + k2 * 20, 40))
        rect(d, 0, 106, W, 74, (95, 95, 100))
        for lx in range(0, W, 30):
            rect(d, lx, 140, 16, 2, (230, 220, 120))
        rect(d, 40, 118, 70, 10, (235, 235, 240))                       # 비행기
        rect(d, 64, 110, 18, 26, (220, 220, 225))
        rect(d, 100, 112, 8, 8, (220, 220, 225))
        rect(d, 46, 120, 4, 3, (70, 110, 150))

    def overlay(self, d, t, cam):
        tag = self.spec.get("tag", "")
        tw = d.textlength(tag, font=font(32))
        rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (150, 30, 30))
        d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))
        cam.text(d, 160, 40, self.spec.get("sign", ""), 24, (40, 50, 70), center=True, shadow=False)


class LabScene(Scene):
    """방역 연구소 실험실: 방역복 연구원, 시험관, 깨진 시험관, 생물위험 표시."""
    focus = (160, 80)

    def draw(self, d, t):
        rect(d, 0, 0, W, H, (205, 215, 210))
        rect(d, 0, 0, W, 8, (170, 180, 175))
        rect(d, 230, 16, 70, 50, (240, 240, 235))                         # 생물위험 표지
        d.polygon([(265, 22), (292, 60), (238, 60)], fill=(250, 200, 40))
        d.ellipse([259, 40, 271, 52], outline=(20, 20, 20), width=2)
        rect(d, 0, 112, W, 12, (160, 165, 170))                           # 실험대
        rect(d, 0, 124, W, 56, (120, 125, 130))
        for i, tx in enumerate(range(150, 220, 10)):                       # 시험관
            c = (200, 60, 60) if i % 2 else (90, 150, 200)
            rect(d, tx, 92, 5, 20, (230, 240, 240))
            rect(d, tx, 102, 5, 10, c)
        rect(d, 146, 110, 70, 3, (100, 100, 110))
        rect(d, 236, 108, 8, 4, (230, 240, 240))                          # 깨진 시험관
        for k2 in range(5):
            rect(d, 232 + k2 * 4, 110 + (k2 % 2), 2, 2, (200, 60, 60))
        draw_char(d, 80, 124, t, False, "hazmat")
        rect(d, 100, 98, 16, 6, (235, 235, 230))                          # 팔
        if int(t * 1.5) % 2 == 0:                                         # 경고등
            d.ellipse([20, 14, 36, 30], fill=(240, 60, 60))

    def overlay(self, d, t, cam):
        tag = self.spec.get("tag", "")
        tw = d.textlength(tag, font=font(32))
        rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (150, 30, 30))
        d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))
        rect(d, VIEW_R - 268, 28, 236, 40, (0, 0, 0))
        d.text((VIEW_R - 256, 32), "ILLUSTRATION", font=font(32), fill=(255, 230, 120))


class HospitalScene(Scene):
    """격리된 병원: 출입통제 테이프, 구급차, 마스크 쓴 사람들."""
    focus = (160, 80)

    def draw(self, d, t):
        vgrad(d, 0, 100, (150, 170, 200), (200, 210, 220))
        rect(d, 60, 8, 200, 76, (235, 235, 230))
        for wx in range(72, 250, 20):
            for wy in (18, 40):
                rect(d, wx, wy, 12, 14, (120, 160, 190))
        rect(d, 150, 0, 20, 14, (235, 235, 230))
        rect(d, 157, 2, 6, 10, (220, 40, 40))
        rect(d, 153, 5, 14, 4, (220, 40, 40))
        rect(d, 140, 62, 40, 22, (150, 190, 210))
        rect(d, 0, 84, W, 96, (130, 130, 135))
        for k2 in range(0, W, 16):                                       # 출입통제 테이프
            c = (250, 210, 40) if (k2 // 16) % 2 == 0 else (20, 20, 20)
            rect(d, k2, 80 + int(math.sin(t * 2 + k2 * 0.1) * 1), 16, 4, c)
        ax = 20 + int(math.sin(t * 0.5) * 4)                              # 구급차
        rect(d, ax, 94, 60, 22, (245, 245, 245))
        rect(d, ax + 46, 98, 12, 8, (150, 200, 230))
        rect(d, ax + 20, 100, 12, 4, (220, 40, 40))
        rect(d, ax + 24, 96, 4, 12, (220, 40, 40))
        if int(t * 4) % 2 == 0:
            rect(d, ax + 26, 90, 8, 4, (60, 120, 255))
        for ax2 in (ax + 8, ax + 44):
            rect(d, ax2, 114, 8, 6, (30, 30, 30))
        for i, px in enumerate((210, 236, 262, 288)):                    # 마스크 쓴 사람들
            rect(d, px - 6, 106, 12, 22, [(60, 90, 140), (140, 60, 60), (70, 70, 70), (90, 130, 90)][i])
            rect(d, px - 4, 96, 8, 10, SKINS[i % 4])
            rect(d, px - 4, 101, 8, 4, (200, 230, 240))

    def overlay(self, d, t, cam):
        tag = self.spec.get("tag", "")
        tw = d.textlength(tag, font=font(32))
        rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, (150, 30, 30))
        d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))
        cam.text(d, 160, 68, self.spec.get("sign", "QUARANTINE"), 24, (220, 40, 40), center=True, shadow=False)


def _tag(d, tag, bg=(30, 40, 90)):
    if not tag:
        return
    tw = d.textlength(tag, font=font(32))
    rect(d, SAFE_X + 32, 28, int(tw) + 32, 44, bg)
    d.text((SAFE_X + 48, 34), tag, font=font(32), fill=(255, 255, 255))


class FlightScene(Scene):
    """대통령 전용기가 태평양을 건너 아시아 순방지로 향한다. 들를지 모르는 곳은 점선 '?'."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 180, (20, 60, 120), (10, 30, 70))
        for y in range(8, 180, 12):                                   # 잔물결
            for x in range((y * 7) % 20, W, 20):
                rect(d, x, y, 4, 1, (30, 75, 135))
        d.polygon([(0, 0), (70, 0), (95, 30), (85, 70), (60, 95), (30, 120), (0, 125)], fill=(70, 120, 70))   # 중국
        d.polygon([(70, 20), (92, 18), (96, 40), (84, 52), (78, 40)], fill=(90, 140, 85))                     # 한반도
        d.polygon([(110, 30), (122, 28), (118, 60), (104, 70)], fill=(90, 140, 85))                           # 일본
        d.polygon([(70, 120), (80, 118), (84, 140), (76, 150)], fill=(90, 140, 85))                           # 필리핀
        d.polygon([(290, 0), (W, 0), (W, 90), (300, 70)], fill=(70, 120, 70))                                 # 미국
        pts = {k: tuple(v) for k, v in self.spec.get("stops", {}).items()}
        for name, (x, y) in pts.items():
            rect(d, x - 2, y - 2, 5, 5, (255, 210, 60))
        maybe = self.spec.get("maybe")
        if maybe and int(t * 2) % 2 == 0:
            d.ellipse([maybe[0] - 5, maybe[1] - 5, maybe[0] + 5, maybe[1] + 5], outline=(255, 80, 80), width=1)
        start, end = (300, 40), tuple(self.spec.get("dest", (60, 95)))
        k = min(1.0, t / max(1.0, self.duration * 0.7))
        for i in range(0, 40):                                         # 지나온 항로 점선
            q = i / 40
            if q > k:
                break
            x = start[0] + (end[0] - start[0]) * q
            y = start[1] + (end[1] - start[1]) * q - math.sin(q * math.pi) * 30
            if i % 2 == 0:
                rect(d, int(x), int(y), 2, 1, (240, 240, 250))
        x = start[0] + (end[0] - start[0]) * k
        y = start[1] + (end[1] - start[1]) * k - math.sin(k * math.pi) * 30
        x, y = int(x), int(y)
        rect(d, x - 9, y - 1, 18, 3, (235, 235, 240))                 # 기체 (왼쪽으로 비행)
        rect(d, x - 9, y + 1, 18, 1, (60, 110, 190))
        d.polygon([(x - 1, y), (x + 4, y - 7), (x + 6, y - 7), (x + 4, y)], fill=(225, 225, 230))
        d.polygon([(x - 1, y + 2), (x + 4, y + 8), (x + 6, y + 8), (x + 4, y + 2)], fill=(225, 225, 230))
        rect(d, x + 7, y - 4, 2, 4, (60, 110, 190))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))
        for name, (x, y) in self.spec.get("stops", {}).items():
            cam.text(d, x + 5, y - 4, name, 28, (255, 230, 120))
        maybe = self.spec.get("maybe")
        if maybe:
            cam.text(d, maybe[0] + 7, maybe[1] - 6, maybe[2], 28, (255, 120, 120))


class SummitsScene(Scene):
    """세 번의 정상회담: 장소가 바뀌며 두 정상이 악수한다."""
    zoomable = False

    def _idx(self, t):
        n = len(self.spec.get("stops", [])) or 1
        return min(n - 1, int(t / max(0.1, self.duration) * n))

    def draw(self, d, t):
        st = self.spec.get("stops", [{}])[self._idx(t)]
        kind = st.get("kind", "singapore")
        if kind == "singapore":                                       # 성조기/인공기가 번갈아 선 무대
            rect(d, 0, 0, W, 120, (150, 30, 40))
            for i, fx in enumerate(range(20, W, 34)):
                draw_flag(d, "us" if i % 2 == 0 else "nk", fx, 18, 26, 70)
        elif kind == "hanoi":                                         # 호텔 회의장: 노란 벽과 커튼
            rect(d, 0, 0, W, 120, (200, 170, 110))
            for cx in range(0, W, 40):
                rect(d, cx, 0, 14, 120, (170, 40, 40))
            draw_flag(d, "us", 110, 20, 40, 26)
            draw_flag(d, "nk", 170, 20, 40, 26)
        else:                                                         # 판문점: 파란 막사 사이 군사분계선
            vgrad(d, 0, 120, (120, 170, 220), (190, 210, 230))
            for bx in (30, 120, 210):
                rect(d, bx, 50, 70, 50, (60, 120, 200))
                rect(d, bx, 46, 70, 6, (220, 220, 225))
            rect(d, 0, 100, W, 20, (170, 160, 140))
        rect(d, 0, 120, W, 60, (60, 50, 50) if kind != "dmz" else (150, 140, 120))
        if kind == "dmz":
            rect(d, 159, 100, 2, 80, (230, 230, 230))                 # 분계선 턱
        sway = int(math.sin(t * 6) * 1)
        draw_char(d, 130, 128, t, False, "trump")
        draw_char(d, 190, 128, t, False, "kim_jong_un")
        rect(d, 146, 112 + sway, 28, 6, (240, 195, 160))              # 맞잡은 손
        rect(d, 158, 111 + sway, 4, 8, (225, 180, 150))

    def overlay(self, d, t, cam):
        st = self.spec.get("stops", [{}])[self._idx(t)]
        _tag(d, st.get("label", ""), (150, 30, 40))


class TruckPlantScene(Scene):
    """트럭 공장 행사: 연단의 트럼프, 옆에 선 로드먼, 뒤로 늘어선 대형 트럭."""
    focus = (160, 80)

    def draw(self, d, t):
        rect(d, 0, 0, W, 120, (70, 75, 85))
        for x in range(0, W, 32):                                     # 철골 지붕
            d.line([(x, 0), (x + 16, 20), (x + 32, 0)], fill=(95, 100, 110))
        rect(d, 0, 20, W, 3, (110, 115, 125))
        for i, tx in enumerate((10, 230)):                            # 보닛 긴 대형 트럭
            c = [(190, 30, 40), (30, 70, 150)][i]
            rect(d, tx, 70, 34, 26, c)
            rect(d, tx + 34, 80, 46, 16, c)
            rect(d, tx + 4, 74, 22, 10, (150, 190, 220))
            rect(d, tx + 78, 82, 4, 12, (200, 200, 205))
            rect(d, tx + 26, 50, 3, 22, (200, 200, 205))
            for wx in (tx + 8, tx + 56):
                d.ellipse([wx, 90, wx + 14, 104], fill=(25, 25, 25))
        rect(d, 0, 104, W, 76, (120, 120, 125))
        rect(d, 100, 20, 120, 14, (20, 30, 70))                       # 현수막
        draw_char(d, 145, 132, t, False, "trump")
        rect(d, 130, 104, 30, 30, (60, 40, 30))                       # 연단
        rect(d, 140, 112, 10, 6, (220, 200, 120))
        rod = {"skin": 3, "hair": 2, "top": [200, 40, 60], "glasses": True}
        draw_person(d, 196, 124, t, False, rod, gesture=False)
        for i, c in enumerate(((255, 80, 160), (80, 220, 255), (255, 220, 60))):   # 알록달록 머리
            rect(d, 189 + i * 5, 92, 4, 4, c)

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))
        cam.text(d, 160, 23, self.spec.get("banner", ""), 28, (255, 255, 255), center=True)
        cam.text(d, 145, 140, "TRUMP", 24, (255, 255, 255), center=True)
        cam.text(d, 196, 140, "RODMAN", 24, (255, 230, 60), center=True)


class LaunchScene(Scene):
    """동해로 발사된 탄도미사일: 이동식 발사대, 짧은 궤적과 긴 '불규칙' 궤적."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 110, (150, 190, 230), (220, 225, 230))
        d.polygon([(0, 110), (0, 70), (40, 80), (90, 95), (110, 110)], fill=(90, 110, 80))   # 내륙 산
        rect(d, 0, 110, 110, 70, (110, 130, 90))
        rect(d, 110, 110, W, 70, (40, 90, 150))
        for y in range(116, 180, 8):
            for x in range(116 + (y % 16), W, 16):
                rect(d, x, y, 5, 1, (70, 120, 175))
        rect(d, 30, 98, 50, 10, (80, 90, 60))                          # 이동식 발사대
        for wx in range(32, 78, 9):
            d.ellipse([wx, 105, wx + 7, 112], fill=(25, 25, 25))
        d.polygon([(40, 98), (70, 74), (74, 77), (46, 100)], fill=(110, 120, 85))
        shots = [(0.15, 120, 22), (0.5, self.spec.get("far", 300), 48)]                      # (시작비율, 착탄 x, 최고고도)
        for i, (s0, x1, hgt) in enumerate(shots):
            k = (t / max(1.0, self.duration) - s0) / 0.3
            if k <= 0:
                continue
            k = min(1.0, k)
            prev = None
            for j in range(int(k * 60) + 1):
                q = j / 60
                x = 72 + (x1 - 72) * q
                y = 74 + (38 * q) - math.sin(q * math.pi) * hgt * (1.6 if i else 1)
                if i == 1 and q > 0.6:                                 # 불규칙 비행
                    y += math.sin(q * 40) * 3
                if prev:
                    d.line([prev, (x, y)], fill=(250, 250, 250), width=1)
                prev = (x, y)
            if k < 1:
                rect(d, int(prev[0]) - 1, int(prev[1]) - 1, 3, 3, (255, 150, 50))
            else:
                r = 4 + int(math.sin(t * 6) * 2)
                d.ellipse([x1 - r, 112 - r // 2, x1 + r, 112 + r // 2], outline=(255, 255, 255))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""), (150, 30, 40))
        cam.text(d, max(22, SAFE_X // 4 + 4), 114, "PYONGYANG AREA", 24, (255, 255, 255))
        for s0, x1, label in ((0.15, 120, self.spec.get("first", "")), (0.5, self.spec.get("far", 300), self.spec.get("second", ""))):
            if t / max(1.0, self.duration) > s0 + 0.3:
                cam.text(d, min(x1, VIEW_R // 4 - 24), 118, label, 28, (255, 230, 60), center=True)
        if t / max(1.0, self.duration) > 0.5:
            cam.text(d, 200, 150, self.spec.get("gap", ""), 28, (255, 255, 255), center=True)


class MuseumScene(Scene):
    """국방 박물관 개관: 거대한 ICBM과 발사차량 앞에 선 김정은 가족."""
    focus = (160, 80)

    def draw(self, d, t):
        rect(d, 0, 0, W, 120, (200, 195, 185))
        for x in range(0, W, 40):                                      # 기둥
            rect(d, x, 0, 8, 120, (225, 220, 210))
        rect(d, 0, 0, W, 10, (170, 30, 40))
        for i in range(4):                                             # 매달린 드론 모형
            dx = 30 + i * 80 + int(math.sin(t + i) * 2)
            rect(d, dx, 18, 16, 3, (90, 95, 90))
            rect(d, dx + 6, 14, 4, 4, (90, 95, 90))
        rect(d, 40, 70, 240, 16, (90, 100, 70))                         # 11축 발사차량
        for wx in range(46, 270, 20):
            d.ellipse([wx, 82, wx + 12, 94], fill=(30, 30, 30))
        rect(d, 60, 52, 200, 16, (70, 80, 60))                          # 미사일 (가로 거치)
        d.polygon([(260, 52), (282, 60), (260, 68)], fill=(70, 80, 60))
        rect(d, 120, 52, 2, 16, (200, 40, 40))
        rect(d, 0, 100, W, 80, (150, 140, 130))
        rect(d, 0, 100, W, 2, (120, 110, 100))
        draw_char(d, 130, 140, t, False, "kim_jong_un")
        wife = {"skin": 0, "hair": 0, "top": [30, 30, 40], "long": True}
        draw_person(d, 170, 142, t, False, wife, gesture=False)
        child = {"skin": 0, "hair": 0, "top": [20, 20, 25], "long": True}
        draw_person(d, 100, 150, t, False, child, gesture=False)

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""), (150, 30, 40))
        cam.text(d, 160, 40, self.spec.get("label", ""), 32, (255, 255, 255), center=True)


class ApecScene(Scene):
    """선전 APEC 회의장 밤 풍경: 마주 볼지 모르는 두 정상, 그 사이의 물음표."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 120, (10, 15, 40), (60, 40, 90))
        rnd = random.Random(5)
        for bx in range(0, W, 18):                                     # 선전 스카이라인
            bh = rnd.randint(30, 90)
            rect(d, bx, 120 - bh, 15, bh, (25, 30, 55))
            for wy in range(122 - bh, 118, 6):
                if rnd.random() < 0.5:
                    rect(d, bx + 3 + rnd.randint(0, 8), wy, 2, 2, (255, 220, 120))
        rect(d, 0, 120, W, 60, (30, 30, 45))
        draw_char(d, 90, 132, t, False, "trump")
        draw_char(d, 230, 132, t, False, "kim_jong_un")

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))
        if int(t * 1.5) % 2 == 0:
            cam.text(d, 160, 48, "?", 160, (255, 210, 60), center=True)



class PowFlightScene(Scene):
    """밤 활주로: 쿠르스크에서 붙잡힌 북한군 포로 두 명이 호송을 받으며 서울행 비행기에 오른다."""
    focus = (160, 80)

    def draw(self, d, t):
        vgrad(d, 0, 100, (8, 10, 30), (30, 35, 70))
        rnd = random.Random(9)
        for _ in range(30):
            rect(d, rnd.randint(0, W), rnd.randint(0, 70), 1, 1, (200, 200, 220))
        rect(d, 0, 100, W, 80, (45, 45, 55))
        for lx in range(0, W, 24):
            rect(d, lx, 104, 3, 2, (255, 210, 90) if int(t * 3 + lx) % 2 else (120, 100, 60))
        rect(d, 170, 66, 130, 22, (210, 210, 220))                     # 수송기 동체
        d.polygon([(300, 66), (316, 50), (318, 66)], fill=(200, 200, 210))
        d.polygon([(220, 76), (260, 76), (240, 96)], fill=(190, 190, 200))
        for wx in range(180, 290, 10):
            rect(d, wx, 72, 5, 4, (60, 80, 110))
        d.polygon([(180, 88), (150, 112), (160, 112), (190, 88)], fill=(150, 150, 160))   # 탑승 계단
        walk = min(1.0, t / max(1.0, self.duration * 0.8))
        for i in range(2):                                              # 모자 쓴 포로 둘 + 호송원
            x = int(40 + walk * 90 + i * 14)
            rect(d, x, 92, 8, 16, (70, 75, 60))
            rect(d, x + 1, 84, 6, 7, (235, 190, 155))
            rect(d, x, 82, 8, 4, (40, 40, 40))
        gx = int(30 + walk * 90)
        rect(d, gx, 90, 9, 18, (30, 35, 50))
        rect(d, gx + 1, 82, 7, 7, (225, 180, 150))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))
        cam.text(d, 160, 20, self.spec.get("route", ""), 32, (255, 230, 120), center=True)


class EmbassyScene(Scene):
    """키이우의 한국 대사관: 태극기 아래 대사가 짐을 들고 차에 오른다(소환)."""
    focus = (160, 80)

    def draw(self, d, t):
        vgrad(d, 0, 110, (150, 170, 190), (200, 205, 210))
        rect(d, 40, 30, 240, 80, (215, 205, 185))
        rect(d, 40, 26, 240, 6, (160, 150, 135))
        for wx in range(56, 270, 26):
            for wy in (42, 70):
                rect(d, wx, wy, 14, 16, (90, 110, 140))
        rect(d, 146, 76, 28, 34, (110, 80, 60))
        rect(d, 158, 0, 2, 30, (200, 200, 200))                          # 깃대 + 태극기
        draw_flag(d, "kr", 160, 2, 30, 20)
        rect(d, 0, 110, W, 70, (120, 120, 125))
        k = min(1.0, t / max(1.0, self.duration * 0.6))
        cx = int(150 + k * 70)
        draw_person(d, cx, 140, t, False, {"skin": 0, "hair": 3, "top": [30, 35, 55], "suit": True}, gesture=False)
        rect(d, cx + 12, 122, 10, 12, (90, 60, 40))                       # 서류가방
        rect(d, 240, 112, 70, 20, (25, 25, 30))                          # 대기 중인 차
        rect(d, 252, 104, 44, 10, (35, 35, 45))
        for wx in (248, 290):
            d.ellipse([wx, 126, wx + 12, 138], fill=(10, 10, 10))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""), (40, 60, 140))
        cam.text(d, 100, 33, self.spec.get("sign", ""), 20, (40, 40, 50), center=True, shadow=False)


class TankerScene(Scene):
    """항구에서 출항하는 유조선: 연료 수출 논란."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 100, (170, 190, 210), (210, 215, 220))
        for cx in range(10, 120, 26):                                     # 항만 크레인
            rect(d, cx, 30, 3, 70, (200, 80, 40))
            rect(d, cx, 30, 22, 3, (200, 80, 40))
        for tx in range(20, 110, 24):                                     # 저장 탱크
            rect(d, tx, 82, 20, 18, (225, 225, 225))
        rect(d, 0, 100, W, 80, (50, 90, 130))
        for y in range(106, 180, 8):
            for x in range((y * 3) % 18, W, 18):
                rect(d, x, y, 6, 1, (80, 120, 160))
        sx = int(130 + (t * 6) % 40)
        d.polygon([(sx, 96), (sx + 150, 96), (sx + 140, 116), (sx + 8, 116)], fill=(150, 40, 40))
        rect(d, sx + 5, 90, 140, 7, (60, 60, 70))
        rect(d, sx + 118, 72, 22, 20, (235, 235, 235))
        rect(d, sx + 126, 64, 6, 8, (40, 40, 40))
        for px in range(sx + 14, sx + 110, 18):
            rect(d, px, 86, 10, 4, (200, 200, 60))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))
        q = self.spec.get("claim", "")
        if q:
            tw = d.textlength(q, font=font(32))
            rect(d, VIEW_R - int(tw) - 64, 28, int(tw) + 32, 44, (150, 30, 40))
            d.text((VIEW_R - tw - 48, 34), q, font=font(32), fill=(255, 255, 255))


class RefineryScene(Scene):
    """밤의 러시아 정유소: 드론이 날아들고 증류탑이 불탄다."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 120, (10, 10, 25), (50, 30, 40))
        rect(d, 0, 112, W, 68, (35, 35, 40))
        for i, (tx, th) in enumerate(((60, 70), (90, 90), (120, 60), (210, 80), (240, 65))):
            rect(d, tx, 112 - th, 12, th, (90, 90, 100))
            for yy in range(112 - th + 6, 112, 10):
                rect(d, tx - 2, yy, 16, 2, (120, 120, 130))
        for bx in (150, 175):
            d.ellipse([bx, 92, bx + 22, 114], fill=(170, 170, 175))
        for k2 in range(8):                                               # 불길과 검은 연기
            ph = (t * 0.7 + k2 / 8) % 1
            sx = 96 + int(math.sin(k2 * 2 + t) * 10)
            sy = int(40 - ph * 40)
            r = 5 + int(ph * 14)
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=lerp((60, 40, 30), (25, 25, 30), ph))
        for k2 in range(6):
            fh = 8 + int(abs(math.sin(t * 7 + k2)) * 10)
            rect(d, 86 + k2 * 4, 22 - fh + 20, 4, fh, (255, 130 + k2 * 18, 40))
        dx = int(W - (t * 50) % (W + 40))                                 # 날아드는 드론
        d.polygon([(dx, 30), (dx + 12, 26), (dx + 12, 34)], fill=(200, 200, 200))
        rect(d, dx + 12, 29, 6, 2, (200, 200, 200))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""), (150, 30, 40))


class PipelineScene(Scene):
    """멈춘 드루즈바 송유관: 국경을 넘는 파이프와 잠긴 밸브, 비어가는 저장 탱크."""
    zoomable = False

    def draw(self, d, t):
        vgrad(d, 0, 100, (160, 185, 210), (215, 220, 215))
        rect(d, 0, 100, W, 80, (120, 140, 90))
        rect(d, 158, 40, 3, 140, (240, 240, 240))                          # 국경선
        for y in range(40, 180, 8):
            rect(d, 158, y, 3, 4, (220, 40, 40))
        rect(d, 0, 110, W, 10, (90, 95, 100))                               # 송유관
        rect(d, 0, 110, W, 2, (130, 135, 140))
        for jx in range(20, W, 40):
            rect(d, jx, 108, 4, 14, (70, 75, 80))
        rect(d, 120, 98, 16, 12, (180, 40, 40))                             # 잠긴 밸브
        rect(d, 116, 94, 24, 4, (180, 40, 40))
        lvl = 1.0 - 0.7 * min(1.0, t / max(1.0, self.duration))
        rect(d, 220, 60, 50, 50, (210, 210, 210))
        rect(d, 224, 64 + int(42 * (1 - lvl)), 42, int(42 * lvl), (40, 40, 40))
        draw_flag(d, self.spec.get("left", "ua"), 40, 50, 36, 24)
        draw_flag(d, self.spec.get("right", "sk"), 200, 20, 36, 24)

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))
        cam.text(d, 58, 78, self.spec.get("left_label", ""), 24, (255, 255, 255), center=True)
        cam.text(d, 218, 48, self.spec.get("right_label", ""), 24, (255, 255, 255), center=True)


class EnvoysScene(Scene):
    """키이우의 대사 회의: 연단의 젤렌스키 앞에 각국 대사들이 앉아 있고, 한 자리에 일본 국기 명패."""
    focus = (160, 70)

    def draw(self, d, t):
        rect(d, 0, 0, W, 120, (60, 70, 100))
        for x in range(0, W, 30):
            rect(d, x, 0, 12, 120, (70, 82, 115))
        draw_flag(d, "ua", 230, 14, 54, 34)
        draw_char(d, 160, 78, t, False, "zelensky")
        rect(d, 142, 62, 36, 26, (110, 80, 60))                          # 연단
        rect(d, 0, 88, W, 92, (50, 45, 45))
        flags = self.spec.get("flags", ["jp", "uk", "fr", "pl", "us", "kr"])
        for i, fx in enumerate(range(18, W, 50)):                        # 뒤돌아 앉은 대사들
            hx = fx + 10
            rect(d, hx - 9, 104, 20, 18, (30, 32, 45))
            rect(d, hx - 6, 94, 13, 12, HAIRS[i % len(HAIRS)])
            kind = flags[i % len(flags)]
            draw_flag(d, kind, hx + 12, 102, 12, 8)
            if kind == self.spec.get("highlight") and int(t * 2) % 2 == 0:
                d.rectangle([hx + 10, 100, hx + 25, 111], outline=(255, 230, 60))

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""))


class ParcelScene(Scene):
    """돌려보낸 훈장: 백색 독수리 훈장이 택배 상자에 담겨 바르샤바로 간다."""
    focus = (160, 80)

    def draw(self, d, t):
        rect(d, 0, 0, W, 120, (205, 200, 190))
        rect(d, 0, 120, W, 60, (150, 110, 80))
        k = min(1.0, t / max(1.0, self.duration * 0.5))
        bx, by = 110, 60
        rect(d, bx, by + 20, 100, 44, (190, 150, 100))                     # 상자
        rect(d, bx, by + 20, 100, 4, (160, 120, 80))
        rect(d, bx + 46, by + 20, 8, 44, (210, 60, 60))                    # 테이프
        lid = int(20 * (1 - k))
        d.polygon([(bx, by + 20), (bx - 14, by + 20 - lid), (bx + 36, by + 20 - lid), (bx + 50, by + 20)], fill=(200, 160, 110))
        my = int(by - 20 + k * 40)                                          # 훈장이 상자 안으로
        if k < 1:
            rect(d, bx + 44, my - 16, 12, 12, (220, 40, 50))
            d.polygon([(bx + 50, my - 2), (bx + 40, my + 10), (bx + 50, my + 20), (bx + 60, my + 10)], fill=(240, 240, 240))
            rect(d, bx + 48, my + 6, 4, 8, (230, 190, 70))
        rect(d, bx + 6, by + 40, 34, 16, (245, 245, 240))                   # 송장
        rect(d, bx + 8, by + 44, 30, 2, (90, 90, 90))
        rect(d, bx + 8, by + 49, 22, 2, (90, 90, 90))
        lx, rx = (100, 174) if self.spec.get("compact") else (24, 250)
        draw_flag(d, "pl", rx, 20, 46, 30)
        draw_flag(d, "ua", lx, 20, 46, 30)

    def overlay(self, d, t, cam):
        _tag(d, self.spec.get("tag", ""), (150, 30, 40))
        cam.text(d, 160, 56 if self.spec.get("compact") else 16, self.spec.get("label", ""), 28, (40, 40, 50), center=True, shadow=False)


SCENES = {"envoys": EnvoysScene, "parcel": ParcelScene, "powflight": PowFlightScene, "embassy": EmbassyScene, "tanker": TankerScene, "refinery": RefineryScene, "pipeline": PipelineScene, "flight": FlightScene, "summits": SummitsScene, "truckplant": TruckPlantScene, "launch": LaunchScene, "museum": MuseumScene, "apec": ApecScene, "airdefense": AirDefenseScene, "airport": AirportScene, "lab": LabScene, "hospital": HospitalScene,
          "protest": ProtestScene, "classroom": ClassroomScene, "xpost": XPostScene, "press": PressScene, "dmz": DMZScene, "koreamap": KoreaMapScene, "un": UNScene, "audit": AuditScene, "call": CallScene, "pump": PumpScene, "quote": QuoteScene, "title": TitleScene, "anchor": AnchorScene, "rally": RallyScene, "map": MapScene,
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


def draw_subtitle(d, scene, t, left=48, width=FW - 96, bottom=652, fs=32, top=None):
    _, e = scene.line_at(t)
    if e is None:
        return
    l = e["line"]
    text = e["clean"]
    n = int(len(text) * min(1.0, (t - e["start"]) / max(0.1, e["adur"] * 0.85)))
    f = font(fs)
    lh = int(fs * 1.25)
    rows = wrap(d, text, f, width - 64)
    box_h = 20 + len(rows) * lh
    y0 = top + lh if top is not None else bottom - box_h
    col = SPEAKER_COLOR.get(l["speaker"], (90, 90, 110))
    rect(d, left, y0, width, box_h, (0, 0, 0))
    rect(d, left, y0, 6, box_h, col)
    lw = d.textlength(l["speaker"], font=f)
    rect(d, left, y0 - lh, int(lw) + 24, lh, col)
    d.text((left + 12, y0 - lh + 2), l["speaker"], font=f, fill=(255, 255, 255))
    pos = 0
    for i, row in enumerate(rows):
        start = text.find(row, pos)
        part = row[:max(0, n - start)]
        pos = start + len(row)
        x = left + 28
        for j, ch in enumerate(part):
            ci = start + j
            hot = any(a <= ci < b for _, a, b, _ in e["emph"])
            if hot:
                rect(d, int(x), y0 + 8 + i * lh, int(d.textlength(ch, font=f)) + 1, lh - 4, (200, 30, 40))
            d.text((x, y0 + 10 + i * lh), ch, font=f, fill=(255, 230, 80) if hot else (255, 255, 255))
            x += d.textlength(ch, font=f)


def active_emph(scene, t):
    for e in scene.timeline:
        for te, _, _, w in e["emph"]:
            if te <= t < te + EMPH:
                return t - te, w
    return None


def draw_stinger(d, dt, word, cy=250, fw=FW):
    """강조 단어 팝업 (튀어나왔다가 자리 잡음)."""
    k = ease(dt / 0.15)
    size = int(48 + 40 * (1 - k)) if dt < 0.15 else 48 + (8 if dt < 0.25 else 0)
    f = font(size)
    txt = word.upper()
    tw = d.textlength(txt, font=f)
    x, y = (fw - tw) / 2, cy - size / 2
    if dt > EMPH - 0.12:
        x += (dt - (EMPH - 0.12)) / 0.12 * fw
    rect(d, int(x) - 24, int(y) - 12, int(tw) + 48, size + 24, (200, 30, 40))
    rect(d, int(x) - 24, int(y) + size + 6, int(tw) + 48, 6, (255, 210, 60))
    d.text((x + 4, y + 4), txt, font=f, fill=(0, 0, 0))
    d.text((x, y), txt, font=f, fill=(255, 255, 255))


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
    beat = 60 / 140
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


def build_audio(scenes, total, hits=False, whoosh=False):
    buf = np.zeros(int(SR * (total + 1)))

    def add(start, sig):
        i = int(start * SR)
        buf[i:i + len(sig)] += sig[: len(buf) - i]

    music = bgm(total) * 0.5
    voice = np.zeros_like(buf)
    t0 = 0.0
    for sc in scenes:
        if isinstance(sc, TitleScene):
            for k, nfreq in enumerate([523, 659, 784, 1047, 784, 1047]):
                add(t0 + 0.2 + k * 0.14, square(nfreq, 0.13, 0.18))
                add(t0 + 0.2 + k * 0.14, square(nfreq / 2, 0.13, 0.10))
        elif whoosh:  # 장면 전환 휙 소리 (기본 꺼짐)
            wt = np.arange(int(SR * 0.25)) / SR
            add(t0, np.random.default_rng(1).normal(0, 1, len(wt)) * np.sin(np.pi * wt / 0.25) * 0.08)
        if isinstance(sc, RallyScene):
            n = int(SR * sc.duration)
            noise = np.convolve(np.random.default_rng(0).normal(0, 1, n), np.ones(40) / 40, "same")
            swell = 0.5 + 0.5 * np.abs(np.sin(np.arange(n) / SR * 1.3))
            add(t0, noise * 0.25 * swell)
        for e in sc.timeline:
            for te, _, _, _ in (e["emph"] if hits else []):   # 강조 '쿵' 효과음
                ht = np.arange(int(SR * 0.35)) / SR
                hit = np.sin(2 * np.pi * (90 - 60 * ht) * ht) * np.exp(-ht * 9) * 0.35
                add(t0 + te - 0.05, hit)
                add(t0 + te - 0.05, square(1320, 0.06, 0.04))
            i = int((t0 + e["start"]) * SR)
            a = e["audio"]
            voice[i:i + len(a)] += a[: len(voice) - i]
        t0 += sc.duration
    # 목소리 나올 때 배경음 낮추기 (더킹)
    k = SR // 10
    env = np.convolve(np.abs(voice), np.ones(k) / k, "same")
    duck = 1 - 0.45 * np.clip(env * 20, 0, 1)
    m = min(len(buf), len(music))
    buf[:m] += music[:m] * duck[:m]
    buf += voice
    buf = np.clip(buf, -1, 1)
    return (buf * 32767).astype(np.int16)


# ---------------------------------------------------------------- 숏폼 (세로)

SW, SH = 1080, 1920
BAND_Y = 520   # 장면이 들어가는 띠 위치 (높이 720)


def compose_short(frame, sc, t, gt, total, em, ep, focus_x):
    if ep.get("layout") == "classic":
        return compose_short_classic(frame, sc, t, gt, total, em, ep, focus_x)
    return compose_short_v2(frame, sc, t, gt, total, ep, focus_x)


def compose_short_v2(frame, sc, t, gt, total, ep, focus_x):
    """숏츠 v2: 첫 2초 '펀치' 문구로 스와이프 막기, 하단 상시 FULL STORY 표시, 끝은 질문으로 롱폼 유도."""
    img = Image.new("RGB", (SW, SH), (12, 16, 40))
    d = ImageDraw.Draw(img)
    cx = int(max(0, min(FW - SW, focus_x - SW // 2)))
    img.paste(frame.crop((cx, 0, cx + SW, FH)), (0, BAND_Y))
    rect(d, 0, BAND_Y - 8, SW, 8, (255, 200, 60))
    rect(d, 0, BAND_Y + FH, SW, 8, (255, 200, 60))
    rect(d, 40, 60, 300, 64, (255, 200, 60))
    d.text((56, 66), "KOWEN NEWS", font=font(52), fill=(15, 20, 50))
    punch = ep.get("punch")
    q = ep.get("question")
    q_on = bool(q) and gt > total - 3.5
    if not (punch and gt < 1.8) and not q_on:
        f = font(88)
        for i, row in enumerate(wrap(d, ep.get("hook", ""), f, SW - 80)[:3]):
            d.text((40, 160 + i * 104), row, font=f, fill=(255, 255, 255), stroke_width=7, stroke_fill=(0, 0, 0))
    if not isinstance(sc, TitleScene):
        draw_subtitle(d, sc, t, left=30, width=SW - 60, fs=48, top=BAND_Y + FH + 40)
    # 첫 1.8초: 위쪽에 큰 펀치 문구 (튀어나오며 시작, 장면은 가리지 않음)
    if punch and gt < 1.8:
        k = ease(gt / 0.18)
        size = int(130 - 40 * (1 - k))
        pf = font(size)
        rows = wrap(d, punch, pf, SW - 60)[:3]
        h = len(rows) * int(size * 1.1)
        y0 = 150 + (BAND_Y - 150 - h) // 2
        rect(d, 0, y0 - 30, SW, h + 60, (200, 30, 40))
        for i, row in enumerate(rows):
            d.text(((SW - d.textlength(row, font=pf)) / 2, y0 + i * int(size * 1.1)), row, font=pf,
                   fill=(255, 255, 255), stroke_width=8, stroke_fill=(0, 0, 0))
    # 상시: 롱폼 버튼 자리를 가리키는 표시
    ya = SH - 210 + int(math.sin(gt * 6) * 8)
    cta = "FULL STORY ▼"
    cf = font(56)
    tw = d.textlength(cta, font=cf)
    rect(d, 40, ya, int(tw) + 40, 76, (200, 30, 40))
    d.text((60, ya + 8), cta, font=cf, fill=(255, 255, 255))
    # 마지막: 답을 안 주고 질문으로 끝내기
    if q_on:   # 위쪽 제목 자리에 질문 카드 (장면은 가리지 않음)
        k = ease((gt - (total - 3.5)) / 0.25)
        qf = font(64)
        rows = wrap(d, q, qf, SW - 120)[:4]
        h = len(rows) * 76 + 50
        y = int(140 + (BAND_Y - 150 - h) // 2 - (1 - k) * 200)
        rect(d, 40, y, SW - 80, h, (255, 200, 60))
        for i, row in enumerate(rows):
            d.text(((SW - d.textlength(row, font=qf)) / 2, y + 25 + i * 76), row, font=qf, fill=(15, 20, 50))
    rect(d, 0, SH - 16, int(SW * gt / total), 16, (255, 200, 60))
    return img


def compose_short_classic(frame, sc, t, gt, total, em, ep, focus_x):
    """가로 장면을 1080 폭으로 잘라 세로 화면 가운데에 넣고, 위에 훅 제목 / 아래에 큰 자막."""
    img = Image.new("RGB", (SW, SH), (12, 16, 40))
    d = ImageDraw.Draw(img)
    cx = int(max(0, min(FW - SW, focus_x - SW // 2)))
    img.paste(frame.crop((cx, 0, cx + SW, FH)), (0, BAND_Y))
    rect(d, 0, BAND_Y - 8, SW, 8, (255, 200, 60))
    rect(d, 0, BAND_Y + FH, SW, 8, (255, 200, 60))
    # 상단: 배지 + 훅 제목
    rect(d, 40, 70, 380, 76, (255, 200, 60))
    d.text((58, 76), "KOWEN NEWS", font=font(64), fill=(15, 20, 50))
    rect(d, 440, 70, 260, 76, (200, 30, 40))
    d.text((462, 82), "BREAKING", font=font(48), fill=(255, 255, 255))
    f = font(80)
    for i, row in enumerate(wrap(d, ep.get("hook", ""), f, SW - 80)[:3]):
        d.text((40, 190 + i * 96), row, font=f, fill=(255, 220, 60), stroke_width=6, stroke_fill=(0, 0, 0))
    if em:
        draw_stinger(d, *em, cy=BAND_Y + 600, fw=SW)
    if not isinstance(sc, TitleScene):
        draw_subtitle(d, sc, t, left=30, width=SW - 60, fs=48, top=BAND_Y + FH + 40)
    # 마지막: 롱폼 유도 문구
    if gt > total - 4.0:
        k = ease((gt - (total - 4.0)) / 0.3)
        y = int(SH - 300 + (1 - k) * 200)
        rect(d, 40, y, SW - 80, 160, (200, 30, 40))
        cta = ep.get("cta", "FULL STORY ON OUR CHANNEL")
        f2 = font(56)
        d.text(((SW - d.textlength(cta, font=f2)) / 2, y + 20), cta, font=f2, fill=(255, 255, 255))
        sub = "▶ Tap the channel name"
        d.text(((SW - d.textlength(sub, font=font(40))) / 2, y + 96), sub, font=font(40),
               fill=(255, 230, 120) if int(gt * 3) % 2 else (255, 255, 255))
    # 진행 바
    rect(d, 0, SH - 16, int(SW * gt / total), 16, (255, 200, 60))
    return img


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
        wf.writeframes(build_audio(scenes, total, ep.get("stinger", False), ep.get("whoosh", False)).tobytes())

    short = ep.get("format") == "short"
    OW, OH = (SW, SH) if short else (FW, FH)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
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
        em = active_emph(sc, t)
        if not ep.get("zoom_cuts", False) or getattr(sc, "zoomable", True) is False or li < 0:
            zi = 0
        elif em:
            zi = 1
        else:   # 줄 안에서도 2.5초마다 컷
            zi = (li + int((t - sc.timeline[li]["start"]) / 2.5)) % 2
        if not ep.get("stinger", False):
            em = None   # 강조 팝업/흔들림 끔 (자막 강조 표시는 유지)
        s, cw, ch = ZOOMS[zi]
        fx, fy = sc.focus_at(t)
        if em and em[0] < 0.2:   # 강조 순간 흔들림
            fx += random.choice((-2, 2))
            fy += random.choice((-1, 1))
        ox = max(0, min(W - cw, fx - cw // 2))
        oy = max(0, min(H - ch, fy - ch // 2))
        img = low.crop((ox, oy, ox + cw, oy + ch)).resize((FW, FH), Image.NEAREST)
        cam = Cam(s, ox, oy)
        d = ImageDraw.Draw(img)
        global SAFE_X, VIEW_R
        SAFE_X = int(max(0, min(FW - SW, (fx - ox) * s - SW // 2))) if short else 0
        VIEW_R = SAFE_X + SW if short else FW
        sc.overlay(d, t, cam)
        if short:
            img = compose_short(img, sc, t, gt, total, em, ep, (fx - ox) * s)
        elif not isinstance(sc, TitleScene):
            if em:
                draw_stinger(d, *em)
            draw_subtitle(d, sc, t)
            draw_ticker(d, ep.get("ticker", []), gt)
        # 장면 전환 화이트 플래시 (어두운 곳에서 눈부셔서 기본 꺼짐)
        if ep.get("flash", False) and t < 0.12 and idx > 0:
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
