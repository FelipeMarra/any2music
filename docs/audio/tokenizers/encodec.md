# EnCodec
Encodec is the audio tokenizer used in [MusicGen](https://musicgen.com/). It is an Auto Encoder (AE) that leverages 1D CNNs to compress informantion and an LSTM to model time. The bottlenec of the AE is further compressed via Residual Vector Quantization (RVQ).

### EnCodec Versions
Encodec has versions for different sample rates as can be seen on its [landing page](https://audiocraft.metademolab.com/encodec.html). But the version used on MusicGen is specially trained to use 32KHz. It's weights can be found on [HuggingFace](https://huggingface.co/facebook/encodec_32khz).

### Example Usage
Please reffer to the EnCodec's test file at [tests/audio/tokenizers/test_encodec.py](../../../tests/audio/tokenizers/test_encodec.py)

### Complementary Material
Interesting video about RVQ that contemplates EnCodec: <br>
https://www.youtube.com/watch?v=Xt9S74BHsvc