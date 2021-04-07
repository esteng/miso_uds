# MISO for Universal Decompositional Semantic Parsing 

## What is MISO? 
MISO stands for Multimodal Inputs, Semantic Outputs. It is a deep learning framework with re-usable components for parsing a variety of semantic parsing formalisms. In various iterations, MISO has been used in the following publications: 

- [AMR Parsing as Sequence-to-Graph Transduction, Zhang et al., ACL 2019](https://www.aclweb.org/anthology/P19-1009/) 
- [Broad-Coverage Semantic Parsing as Transduction, Zhang et al., EMNLP 2019](https://www.aclweb.org/anthology/D19-1392/) 
- [Universal Decompositional Semantic Parsing, Stengel-Eskin et al. ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.746/) 
- [Joint Universal Syntactic and Semantic Parsing, Stengel-Eskin et al., TACL 2021](#TODO) 

If you use the code in a publication, please do cite these works. 

## What is Universal Decompositional Semantics? 
[Universal Decompositional Semantics (UDS)](http://decomp.io/projects/decomp-toolkit/) is a flexible semantic formalism built on [English Web Treebank Universal Dependencies](https://universaldependencies.org/en/overview/introduction.html) parses. 
UDS graphs are directed acyclic graphs on top of UD parses which encode the predicate-argument structure of an utterance. 
These graphs are annotated with rich, scalar-valued semantic inferences obtained from human annotators via crowdsourcing, encoding speaker intuitions about a variety of semantic phenomena including factuality, genericity, and semantic proto-roles. 
More details about the dataset can be found in the following paper: [The Universal Decompositional Semantics Dataset and Decomp Toolkit, White et al., LREC 2020](https://www.aclweb.org/anthology/2020.lrec-1.699/) and at [decomp.io](http://decomp.io/projects/decomp-toolkit/). 

## What is UDS Parsing?  
UDS parsing is the task of transforming an utterance into a UDS graph, automatically. 
Using the existing dataset and the MISO framework, we can learn to parse into UDS. This is a particularly challenging parsing problem, as it involves three levels of parsing
1. Syntactic parsing of the utterance into UD 
2. Parsing the utterance into the UDS graph structure
3. Annotating the graph structure with UDS attributes

## MISO overview 
MISO builds heavily on [AllenNLP](https://github.com/allenai/allennlp), and so many of its core functionalities are the same. 

## Installation 
Using conda, all required libraries can be installed by running: 
- `conda create --name miso python=3.6`
- `conda activate miso`
- `pip install -r requirements.txt`

## Useful scripts 
`experiments/decomp_train.sh` has several functions for training and evaluating UDS parsers via the command-line. This script is used for `DecompParser` models, trained and evaluated without UD parses. Specifically:
- `train()` trains a new model from a configuration, saving checkpoints and logs to a directory specified by the user. If the directory is non-empty, an error will be thrown in order to not overwrite the current model.
- `resume()` is almost identical to `train` except that it takes a non-empty checkpoint directory and resumes training. 
- `eval()` evaluates the structure of the produced graphs with the S-metric, with unused syntax nodes included in the yield of each semantic head. It requires the user to specify which data split to use (test or dev). The outcome of the evaluation is stored in ${CHECKPOINT_DIR}/${TEST_DATA}.synt_struct.out.
- `eval_sem()` is identical to `eval` except that it computes the S-metric for semantics nodes only.
- `eval_attr()` evaluates the S-score with both syntactic nodes and attributes included. 
- `spr_eval()` computes the aggregate Pearson score across all nodes and edges, storing the outcome in ${CHECKPOINT_DIR}/${TEST_DATA}.pearson.out

If training/evaluating a model with syntactic info, a similar script is used: `syntax_experiments/decomp_train.sh`. This script has the same functions as `experiments/decomp_train`, but also contains: 
- `conllu_eval()`, which evaluates the micro-F1 LAS and UAS of predicted and gold UD parses
- `conllu_predict()` produces a CoNLL-U file with the predicted UD parse, which can be scored against a reference file using 3rd party tools. 
- `conllu_predict_multi()` which produces a similar file but from multilingual data (see [multilingual experiments](#TODO) for more)
- `conllu_predict_ai2()` produces a file for the PP-attachement dataset. 

## Configurations 
MISO model instantiation happens via a config file, which is stored as a `.jsonnet` file. Config files are stored in `miso/training_config`, and are organized around 3 broad axes: 
- `lstm` vs `transformer` based models
- separate or joint models, where joint models do syntactic and semantic parsing jointly
- `semantics_only` or `with syntax` models, where `semantics only` models do not include non-head syntactic tokens in the yield of a semantic node, while `with_syntax` models do.  

Models also typically use GloVe embeddings, which need to be downloaded by the user and specified in the config file. 
The config files specify the registered names of classes, which are then instantiated with their options by AllenNLP. 
For example, a DecompDatasetReader is registered as a subclass of `allennlp.data.dataset_readers.dataset_reader` in `miso/data/dataset_readers/decomp.py` with the name `decomp` using the following line: 

```
@DatasetReader.register("decomp") 
```

and then instantiated from a config file with the following specification: 

```
"dataset_reader": {
    "type": "decomp",
    "drop_syntax": true,...
```

This method of setting and manipulating configuration options lets us save configurations easily and saves us from having excessively long commandline arguments. 
Furthermore, because the configs are `.jsonnet` files, we can set and re-use variables in them (e.g. paths to embeddings, etc.) 
The registration method used by allennlp and MISO also lets us easily extend classes to add new models and functionalities. 

## Models 

### UDS Baselines 
- [DecompParser](miso/models/decomp_parser.py) is an LSTM-based UDS-only parsing model similar to the model presented [here](https://www.aclweb.org/anthology/2020.acl-main.746/). It has an encoder and a decoder. 
- [DecompTransformerParser](miso/models/decomp_transformer_parser.py) is a new Transformer-based variant of the `DecompParser` and inherits from it.  
### UD Baselines 
- [DecompSyntaxOnlyParser](miso/models/decomp_syntax_only_parser.py) is an LSTM-based biaffine parser, similar to the one presented [here](https://arxiv.org/abs/1611.01734). It only has an encoder (no decoder) and cannot do UDS parsing, just UD parsing.
- [DecompTransformerSyntaxOnlyParser](miso/models/decomp_transformer_syntax_only_parser.py) the same model but with a Transformer encoder instead of an LSTM.
### Joint syntax-semantics models 
- [DecompSyntaxParser](miso/models/decomp_syntax_parser.py) can do joint UDS and UD parsing with an LSTM encoder/decoder, following one of three strategies: 
    - `concat-before` linearizes the UD parse and concatenates it before the linearized UDS graph
    - `concat-after` concatenates the UD tree after the UDS graph. Both the concatation strateiges are sub-optimal, since the UD formalism is lexicalized, so the model would need to learn to perfectly reconstruct the input tokens in a shuffled order, which is unnecessary. 
    - `encoder` puts a biaffine UD parser on top of the encoder in the encoder-decoder framework for UDS parsing. This model is similar to a `DecompSyntaxOnlyParser` but additionally has a decoder which performs UDS parsing. 
    - `intermediate` is almost the same as the `encoder` model, except that the output of the biaffine parser is re-encoded and passed to the decoder, making the decoder explicitly syntax-aware.  
- [DecompTransformerSyntaxParser](miso/models/decomp_transformer_syntax_parser.py) can do joint UDS and UD parsing with a Transformer encoder/decoder, following one of three strategies: 

### Transformer changes 
UDS has far fewer training examples than most tasks that transformers are applied to. Accordingly, we adopt a number of changes described in [Transformers without Tears: Improving the Normalization of Self-Attention](https://arxiv.org/abs/1910.05895) to adapt them to this low-data regime. 
These changes include:
- pre-normalization (swapping the order of the LayerNorm) 
- scaled initialization 
- smaller warmup rate 

### Contextualized encoders 
Currently, two different contextualized encoders can be used with MISO: BERT and XLM-Roberta (XLM-R). These are specified in config files under `bert_encoder`. Note that, if using a contextualized encoder, the appropriate tokenizer must also be set. BERT can be used by setting the `type` of the encoder to `seq2seq_bert_encoder`, while XLM-R encoders are registered under `seq2seq_xlmr_encoder`. 

## Training 
Training a model can be done by using the script in experiments, or through AllenNLP (since all models here inherit from AllenNLP's `Model` class). For example, to train a joint UD-UDS transformer model with encoder-side biaffine parsing, we would use the following command: 

```
mkdir -p models/encoder 
python -um allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.training \
    --include-package miso.metrics \
    -s models/encoder/ckpt \
    miso/training_config/transformer/with_syntax/encoder.jsonnet
```

The weights, logs, and metrics will be saved to `models/encoder/ckpt`. 
The `--include-package` flags here tell AllenNLP where to look for the registered subclasses which are specified in the `encoder.jsonnet` config file. 
Various metrics and training progress will be logged to `stdout.log`. Any errors will appear in `stderr.log`.  
At the end of training, an archive called `model.tar.gz` containing the following files should be created: `weights.th`, `config.json`, `vocabulary/*`.
If training ends early for some reason and you want to evaluate the model anyway, you can create this file by accessing the model checkpoint directory and inputting 
```
cp best.th weights.th 
tar -czvf model.tar.gz weights.th config.json vocabulary
```

## Testing 
Testing is a bit more complicated, since there are many metrics to consider. 

### Metrics 
#### S metric
     This metric was designed for UDS graphs as an extension of Smatch. It can measure the alignment between a gold and predicted graph, including attribute scores. It is called via `miso.commands.s_score eval` 
```
    python -m miso.commands.s_score eval \
    models/encoder/ckpt/model.tar.gz dev \
    --predictor "decomp_syntax_parsing" \
    --batch-size 32 \
    --beam-size 1 \
    --use-dataset-reader \
    --save-pred-path models/encoder/pred_dev_graphs.pkl\
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
```
As is, this version of S score is equivalent to Smatch. If attribute scores should be included in the overall score, the `--include-attribute-scores` flag should be set. 
If the `--semantics-only` flag is set, it will evaluate the score against just the semantics nodes, ignoring syntactic non-head nodes (see [Section 3](https://www.aclweb.org/anthology/2020.acl-main.746.pdf)).

#### Attribute Metrics
Under a forced ("oracle") decode of the graph structure, we can measure Pearson's $\rho$ between the predicted and gold attribute values for UDS node and edge attributes. 
This can be done by running `s_score.py` command with the `spr_eval` action, as done in the `spr_eval()` function in `syntax_experiments/decomp_train.sh`.
That command will produce a json file which has the predicted and gold edge and node attributes per node and edge.  
Running that json file through `python -m miso.commands.pearson_aggregate` will produce an average Pearson score for the whole system, averaged across all attributes. 

The script also produces an attribute F1 score. Recall that attributes are bounded on [-3, 3], typically with a midpoint at 0. We use the midpoint to binarize the values, computing the F1 score against the binarized gold attributes. In practice, we tune the threshold on the development set. 

#### UD Metrics
UD metrics used are unlabeled and labeled attachment score (UAS/LAS). These are standard UD metrics. 
The official metrics can be computed by running `miso/commands/s_score.py` with the `conllu_predict` argument as done in the `conllu_predict()` function in `syntax_experiments/decomp_train.sh`. 
This will produce a predicted `.conllu` file that can be evaluated against the gold file using the official evaluation script, which for convenience is included in `miso/metrics/conllu.py`. 
However, for debugging, a micro-averaged version of the conllu score can also be computed using the `conllu_eval()` function. 

#### UD data
EWT UD data is included in the repo in `data/UD/EWT` and `data/UD/EWT_clean`. The former directory has the official English UD data, while the latter has the filtered version described in the paper, where sentences without semantic graphs are filtered out. 

    
