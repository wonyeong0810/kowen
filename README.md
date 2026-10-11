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
