import logging
import time
from pprint import pprint

import torch

from fairseq.hub_utils import GeneratorHubInterface

start = time.monotonic()

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# https://pytorch.org/hub/pytorch_fairseq_translation/#english-to-french-translation
# Load an En-Fr Transformer model trained on WMT'14 data :
en2fr = torch.hub.load(
    "pytorch/fairseq", "transformer.wmt14.en-fr", tokenizer="moses", bpe="subword_nmt"
)

with open("output/en2fr_args", "wt") as out:
    pprint(vars(en2fr.args), stream=out)

# Clone into local fairseq
# logging.info(en2fr)
model_clone = GeneratorHubInterface(en2fr.args, en2fr.task, en2fr.models)
model_clone.models.load_state_dict(en2fr.models.state_dict())
en2fr = model_clone

# Use the GPU (optional):
en2fr.cuda()

# Translate with beam search:
fr = en2fr.translate("Hello world!", beam=5)
assert fr == "Bonjour à tous !"

# Manually tokenize:
en_toks = en2fr.tokenize("Hello world!")
assert en_toks == "Hello world !"

# Manually apply BPE:
en_bpe = en2fr.apply_bpe(en_toks)
assert en_bpe == "H@@ ello world !"

# Manually binarize:
en_bin = en2fr.binarize(en_bpe)
assert en_bin.tolist() == [329, 14044, 682, 812, 2]

# Generate five translations with top-k sampling:
fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
assert len(fr_bin) == 5

# Convert one of the samples to a string and detokenize
fr_sample = fr_bin[0]["tokens"]
fr_bpe = en2fr.string(fr_sample)
fr_toks = en2fr.remove_bpe(fr_bpe)
fr = en2fr.detokenize(fr_toks)
assert fr == en2fr.decode(fr_sample)

logging.info("en_bin".rjust(20) + "   %s", en_bin)
for k in fr_bin[0].keys():
    logging.info(k.rjust(20) + "   %s", fr_bin[0][k])

# Translate with beam search:
# https://github.com/pytorch/fairseq/tree/master/examples/translation#example-usage-torchhub
fr = en2fr.translate(["Hello world!", "Hello world!"], beam=5)
assert fr == ["Bonjour à tous !", "Bonjour à tous !"]

# no list for tokenize/apply_bpe/binarize

# Generate five translations with top-k sampling:
en_bin = [en_bin, en_bin]
fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
assert len(fr_bin) == len(en_bin)
assert all(len(i) == 5 for i in fr_bin)

assert time.monotonic() - start < 20
