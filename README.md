# MOSAIC Any2Music
MOSAIC Any2Music (MAM) is a library for multimodal encoder-decoder model components with focus on music generation.

## Installation
```bash
git clone https://github.com/FelipeMarra/any2music.git
cd any2music
python3 -m venv env
source env/bin/activate
python3 -m pip install -e .
```

## Testing
From the repositorie's root directory, run:
```bash
python3 -m pytest
```
The `-s` flag can be used to show the prints inside the tests functions

## Available Components
### Audio Tokenizers
* [EnCodec](docs/audio/tokenizers/encodec.md)
* [DAC (Improved RVQGAN)](docs/audio/tokenizers/dac.md)

### Text Encoders
* [T5](docs/text/encoders/t5.md)

### Audio Decoders
* [MusicGen-like](docs/audio/decoders/musicgen.md)