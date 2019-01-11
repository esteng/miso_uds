# Stog: the String-to-Graph project

This project is based on python 3.6 and the newest version of PyTorch.

## Baselines

- RNN Seq2Seq Model (see [this](https://gitlab.hltcoe.jhu.edu/research/mt-ie/tree/copy) for details) 
- (Optional) Transition-based Semantic Parser

## Running

Example of training (on clsp grid):

`CUDA_VISIBLE_DEVICES=`free-gpu` python -u -m stog.commands.train params/lm.yaml`


## Stage-1 AMR Parsing Timeline

- [x] Build a UD parser using the deep biaffine network.
    - ETA: 9/24 - 10/14
- [x] Build attention-based Seq2Seq for AMR concept prediction. 
    - ETA: 10/15 - 11/12
- [ ] Build the new self-copy mechanism.
    - ETA: 11/12 - 11/30
- [ ] Joint train AMR concept prediction and edge prediction.
    - ETA: 12/1 - 12/15
- [ ] Add AMR-specific pre- and post-processing
    - ETA: 12/1 - 12/30

## Thoughts about AMR

See this [issue](https://gitlab.hltcoe.jhu.edu/szhang/stog/issues/15).

## Related Work

- Deep Biaffine
    - [ICLR2017](https://arxiv.org/pdf/1611.01734.pdf)
    - [CoNLL2017](https://web.stanford.edu/~tdozat/files/TDozat-CoNLL2017-Paper.pdf)
    - [CoNLL2018](http://universaldependencies.org/conll18/proceedings/pdf/K18-2016.pdf)
- [NeuroNLP](https://github.com/XuezheMax/NeuroNLP2)
- [SemEval 2015 Task 18](http://aclweb.org/anthology/S15-2153)
    - [English DM](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1956)
- [CoNLL 2018 Shared Task](http://universaldependencies.org/conll18/)
    - Evaluation Script
- [Spider](https://yale-lily.github.io/spider)

## Data

### [UDS Data](https://gitlab.hltcoe.jhu.edu/research/mt-ie/blob/copy/README.md#uds-data-cross-lingual-semantic-parsing-w-factuality-and-sprs)