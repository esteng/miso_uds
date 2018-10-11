# Stog: the String-to-Graph project

This project is based on python 3.6 and the newest version of PyTorch.

## Baselines

- RNN Seq2Seq Model (see [this](https://gitlab.hltcoe.jhu.edu/research/mt-ie/tree/copy) for details) 
- (Optional) Transition-based Semantic Parser

## Stage-1 Timeline

1. Build a UD parser using the self-attention network.
    - ETA: 9/24 - 10/14
    - [Progress](docs/progress/stage1.1.md)
2. Build a neural PredPatt by adapting the UD parser. 
    - ETA: 10/15 - 11/4
    - [Progress](docs/progress/stage1.2.md)
3. Beat the state of the art in SPR and factuality.
    - ETA: 11/5 - 11/18
    - [Progress](docs/progress/stage1.3.md)
4. Write the paper.
    - ETA: 11/19 - NAACL ddl

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