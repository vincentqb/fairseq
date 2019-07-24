# Finetuning RoBERTa on Winograd Schema Challenge (WSC) data

### 1) Download the WSC data from the SuperGLUE website:
```bash
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip
unzip WSC.zip

# we also need to copy the RoBERTa dictionary into the same directory
wget -O WSC/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
```

### 2) Fine-tune over the provided training data:

The following command can be used to fine-tune RoBERTa with the provided WNLI
data. Note that there is high variance in the results. For our submission we
swept over the hyperparameters listed below, including the random seed. Out of
100 runs we chose the best 7 models and ensembled them. The released
`roberta.large.wsc` model corresponds to the best of the 7 models according to
the development set accuracy.

```bash
TOTAL_NUM_UPDATES=2000  # Total number of training steps.
WARMUP_UPDATES=250      # Linearly increase LR over this many steps.
LR=2e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=16        # Batch size per GPU.
SEED=1                  # Random seed.
ROBERTA_PATH=/path/to/roberta/model.pt

# we use the --user-dir option to load the task and criterion
# from the examples/roberta/wsc directory:
FAIRSEQ_PATH=/path/to/fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/wsc

cd fairseq
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train WSC/ \
  --restore-file $ROBERTA_PATH \
  --reset-optimizer --reset-dataloader --reset-meters \
  --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  --valid-subset val \
  --fp16 \
  --user-dir $FAIRSEQ_USER_DIR \
  --task wsc --criterion wsc --wsc-cross-entropy \
  --arch roberta_large --bpe gpt2 --max-positions 512 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
  --lr-scheduler polynomial_decay --lr $LR \
  --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
  --max-sentences $MAX_SENTENCES \
  --max-update $TOTAL_NUM_UPDATES \
  --log-format simple --log-interval 100
```

The above command assumes training on 4 GPUs, but you can achieve the same
results on a single GPU by adding `--update-freq=4`.

### 3) Evaluate
```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('checkpoints/checkpoint_best.pt')
roberta.disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
# True
```
