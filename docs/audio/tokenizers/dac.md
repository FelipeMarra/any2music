# DAC
Dac is described on the paper [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546). It improves upon [Encodec](encodec.md) but presenting a new activation funtion, improved codebook usage and better adversarial training. The paper's implementation can be found [here](https://github.com/descriptinc/descript-audio-codec).

### Example Usage in Any2Music
Please reffer to the test file at [tests/audio/tokenizers/test_dac.py](../../../tests/audio/tokenizers/test_dac.py) for usage example. There is also an integration test using it with musicgen at [tests/audio/decoders/test_musicgen.py](../../../tests/audio/decoders/test_musicgen.py)

### Complementary Material
You can check this [notion page](https://app.notion.com/p/DAC-Improved-RVQGAN-37f58111e81480838668dfb71582e520?source=copy_link) about it, that contemplates an example on how it works better for NES music.