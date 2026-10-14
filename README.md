# KOWEN NEWS

A generator for news videos presented by pixel-art characters.

- An anchor (male voice) introduces the story from the newsroom.
- A reporter (female voice) explains the details over pixel reenactment scenes.
- The other speakers (passers-by, an expert, overseas reactions) each get their own voice.
- Each line's length follows its synthesized audio. Zoom cuts, flash transitions, and a chiptune BGM keep the pace up.

## Setup

```bash
pip install pillow numpy sherpa-onnx soundfile     # ffmpeg is also required
mkdir -p ~/models && cd ~/models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
tar xjf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
```

The TTS is Supertonic 3, which runs offline and supports 31 languages including English and Korean.
Speaker `sid` 0-4 are female voices and 5-9 are male. If the model lives somewhere else, set `KOWEN_TTS_MODEL`.

## Run

```bash
python3 kowen_news.py episodes/trump_ocean.json   # -> out/trump_ocean.mp4 (about 5 min)
```

## Making a new episode

Copy `episodes/trump_ocean.json` and change the content.

- `lang`: TTS language (`en`, `ko`, ...)
- `flash` / `whoosh`: white flash and whoosh sound on scene changes (both off by default because the flash hurts eyes in the dark)
- `zoom_cuts` / `stinger`: zoom cuts, and the pop-up caption with camera shake on emphasized words (both off by default; emphasized words are still highlighted in the subtitles)
- `voices`: per-speaker voice, as `{"sid": 0-9, "speed": 1.2}`
- `scenes[].lines`: `{"speaker": "...", "text": "..."}`. No duration is needed; it comes from the audio.

| type | What it shows |
|------|---------------|
| `title` | Opening sequence |
| `anchor` | Newsroom anchor (`headline`) |
| `rally` | Campaign rally with a Trump-style character and crowd (`banner`, `podium`) |
| `map` | Ocean renaming on a map (`old`, `new`) |
| `signboard` | Road signs showing name changes (`title`, `items[{old,new,year}]`) |
| `meeting` | International organization meeting (`sign`) |
| `reaction` | Reactions from people abroad (`panels[{speaker,flag,...}]`, flags: uk/fr/br/ca) |
| `chart` | Poll bar chart (`title`, `bars`) |
| `interview` | Street or studio interview (`bg`: street/studio, `person`, `person2`) |

## Shorts guidelines (from channel analytics)

- Target **30–40 seconds**.
- **0–2s:** strong hook. Use `punch` for the first frame and open with the most striking line.
- **Middle:** the key facts, changing scenes every few seconds.
- **End:** finish on a question to viewers (`question`, plus a spoken line asking them to comment). Don't give away the full story, so the long-form video stays worth watching.
- Use the default (v2) Shorts layout. Set "Related video" to the long-form on upload.

## Thumbnail guidelines

- No question marks or exclamation points in thumbnail text.
- When a claim is unconfirmed, use a word like "SCARE" instead of a "?" so the text doesn't state it as fact.

## Long-form script guidelines

- Don't use time-of-day phrases. No "evening news", no "Good evening", no "good night".
  - Opening: "Welcome to KOWEN News."
- Closing: end with a call to action.
  - "If you found this helpful, please like and subscribe for more. This has been KOWEN News. Thanks for watching."
- The title card shows only the KOWEN NEWS logo (no subtitle).

## Scene guidelines

- Reuse the frame templates freely: `anchor`, `quote`, `signboard`, `chart`, `xpost`, title card.
- Reenactment scenes (what actually happened: the place, the event, the people) must be drawn new for every story. Do not recycle another episode's situation scene (e.g. the White House briefing, the Korea map, the generic meeting room); add a new scene class for the event instead.

## Voice pacing (audience is mostly 40+)

- From the next episode on, speak slower: long-form Anchor and Reporter `speed` 1.1 (was 1.25), Shorts 1.15 (was 1.3).
- Slower speech makes episodes longer, so long-form scripts need fewer lines to stay in the 5-15 minute range.
