# KOWEN NEWS

픽셀 캐릭터가 뉴스를 전하는 영상 생성기.

- 뉴스룸에서 앵커가 오늘의 주제를 소개
- 기자는 목소리(자막)로만 등장, 픽셀 재연 화면 위에서 뉴스를 자세히 설명
- 하단 뉴스 티커, 말소리 효과음, 오프닝 징글 자동 생성

## 실행

```bash
pip install pillow numpy   # ffmpeg 필요
python3 kowen_news.py episodes/trump_ocean.json   # -> out/trump_ocean.mp4
```

## 새 에피소드 만들기

`episodes/*.json`을 복사해서 대사만 바꾸면 됩니다. 장면 타입:

| type | 내용 |
|------|------|
| `title` | KOWEN NEWS 오프닝 (`duration`) |
| `anchor` | 뉴스룸 앵커 (`headline`, `lines`) |
| `rally` | 유세 현장 + 트럼프 캐릭터 + 관중 (`banner`, `podium`, `lines`) |
| `map` | 지도 위 바다 이름 바꾸기 연출 (`old`, `old_ko`, `new`, `new_ko`, `lines`) |

`lines`의 각 항목은 `{"speaker": "앵커" | "기자", "text": "...", "dur": 초}`.
새 장면 타입은 `kowen_news.py`에 `Scene`을 상속한 클래스를 만들고 `SCENES`에 등록하면 됩니다.
