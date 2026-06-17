# MOSAIC Any2Music
MOSAIC Any2Music (MAM) is a library for multimodal encoder-decoder model components with focus on music generation.

## Available Components
### Audio Tokenizers
* [EnCodec](docs/audio/tokenizers/encodec.md)
* [DAC (Improved RVQGAN)](docs/audio/tokenizers/dac.md)

### Text Encoders
* [T5](docs/text/encoders/t5.md)

### Audio Decoders
* [MusicGen-like](docs/audio/decoders/musicgen.md)

## Installation
```bash
conda create -n any2music python=3.10.12
conda activate any2music
git clone https://github.com/FelipeMarra/any2music.git
python3 -m pip install -e ./any2music --extra-index-url https://download.pytorch.org/whl/cu126
```

## Testing
From the repositorie's root directory, run:
```bash
python3 -m pytest
```
The `-s` flag can be used to show the prints inside the tests functions