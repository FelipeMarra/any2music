import random
import typing as tp

import torch
from transformers import T5EncoderModel, T5Tokenizer  

from ...base.conditioner import BaseConditioner

import warnings
import logging
logger = logging.getLogger(__name__)

# From https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/modules/conditioners.py#L422
class T5Conditioner(BaseConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = [
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
        "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
        "google/flan-t5-xl", "google/flan-t5-xxl"
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(
            self, name: str, output_dim: int = -1, device: str = 'cuda', finetune: bool = False,
            dtype: torch.dtype = torch.bfloat16, word_dropout: float = 0.,
            # normalize_text: bool = False
        ):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"

        self.d_model = self.MODELS_DIMS[name]

        super().__init__(self.d_model, output_dim)

        self.output_dim = output_dim
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout

        logger.info(f"T5 will be evaluated with autocast as {dtype}")
        self.autocast = torch.autocast(device_type=self.device, dtype=dtype)
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)

        if finetune:
            self.t5 = t5
        else:
            # TODO:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__['t5'] = t5.to(self.device)

        # TODO: Do we need this?
        # self.normalize_text = normalize_text
        # if normalize_text:
        #     self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]], max_length:int=128) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        # if self.normalize_text:
        #     _, _, entries = self.text_normalizer(entries, return_text=True)

        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(
            entries, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=max_length,       # Force all batches to this exact length
            truncation=True       # Cut off anything longer than 128
        ).to(self.device)

        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]):
        mask = inputs['attention_mask']

        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state

        if self.output_dim > -1:
            embeds = self.output_proj(embeds.to(self.output_proj.weight))

        embeds = (embeds * mask.unsqueeze(-1))

        return embeds, mask