# MusicGen
[MusicGen](https://musicgen.com/) is a encoder-decoder model to generated text-conditioned music in raw audio formats. It leverages [T5](../../text/encoders/t5.md) to encode text and [Encodec](../tokenizers/encodec.md) to tokenized the audio. The Encodec tokens are modeled in parallel using a delay pattern.

The original implementation of MusicGen is part of the [Audiocraft](https://github.com/facebookresearch/audiocraft) project, which is complex. The whole motivation behind this repository is to provide a simpler implementation of it.

### Example Usage
Please reffer to the test file at [tests/audio/decoders/test_musicgen.py](../../../tests/audio/decoders/test_musicgen.py) for usage example.