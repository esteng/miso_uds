# Readme for running UD experiments

## current UD/UDS experiments 
### English UD parsing and English UDS parsing: here
Since UD EWT and UDS have parallel, I'm adding a parsing objective to the UDS parsing setup, which is done in multiple steps. The relevant configs are all in `miso/training_config/interface/`, which has 4 subdirs

- lstm: lstm-based configs
- lstm_sem: lstm configs for semantics-only on the UDS side
- transformer: transformer configs
- transformer_sem

Inside each subdir, there are different files (unfortunately with some different naming conventions) but hopefully with informative-enough names. The settings in general are: 
- biaffine only: this is to set a baseline of the performacne of the UD parser on the input features (BERT, GLoVE) without any UDS signal, usually named something like "decomp_syntax_only.jsonnet" 
- encoder-side: adds a UD objective to the loss, but doesn't provide explicit info to the decoder about the graph, e.g. `miso/training_config/interface/lstm/decomp__syntax_semantics_encoder.jsonnet`
- intermediate: adds the UD objective and re-encodes the predicted graph and sends that to the decoder, e.g. `miso/training_config/interface/lstm/decomp_lstm_intermediate.jsonnet`

## Current Multilingual experiments 
The setup here is different, since there's no parallel data here. What I'm doing is training separate models and then comparing them with models initialized by training on the other task. So for example, I'm training a French UD parsing model from scratch, and comparing that with a French UD parsing model where the compatible weights are initialized by weights learned on English UDS parsing. The settings here are 
- baseline: train a UD parsing model in the relevant language using XLMR features, from scratch with only the UD objective
- encoder: intialize the weights with weights learned on UDS parsing with the encoder-side model 
- intermediate: initalize with weights learned by the UDS intermediate model 

All the UDS models (encoder, intermediate) are trained on XLMR features only, so their performance is far lower than the others trained on BERT + GloVE. The transfer also goes the other way, where I'm training a multlingual UD parsing model on the concatenation of all the UD data in all the languages, and then transferring that to the UDS parsing model. Some notes here are: 
- because the transformer gets the best UAS/LAS performance, I am only training transformer models right now 
- the code to load weights from the other model is in `miso/models/transduction_base.py:load_partial()` and only loads in weights if the shapes match, otherwise warning the user that it can't match the weights, so it has to be used carefully or it might not load the weights at all and just randomly init them, but still run. 
- the dataset reader inherits from the AllenNLP multilingual UD reader, which can read in multiple languages simultaneously, but the data needs to be organized in a particular way. The organization that works best I think is:

```all_data
|
| train
    | 
    {lang_code}-universal.conllu
| dev 
    | 
    {lang_code}-universal.conllu
| test
    | 
    {lang_code}-universal.conllu
```

where {lang_code} is replaced with the 2-letter code in the "languages" field in the config. For single languages, that would be a list with one element, but for multilingual models you can just add in more elements into the list. 

The relevant configs here are: 
- `miso/training_config/ud_parsing/transformer/`
- `miso/training_config/ud_parsing/transformer_pretrained/`

where the naming convention is a bit more self-explanatory I think. 

## Submission scripts
I have a bunch of tools for training and decoding models. The relevant files/functions are: 
- `syntax_experiments/decomp_train.sh`
    - `train()` trains a new model 
    - `conllu_predict()` predicts a dev or test .conllu file 
    - `conllu_eval()` computes an estimate (micro-averaged) of UAS/LAS given the dev set 

- `grid_scripts/train_multilang.sh` and `grid_scripts/train_multilang_pretrained.sh` submit training jobs for all of the UD languages, and are good sources for an example of how to train the models 
- `grid_scripts/decode_multilang.sh` decodes .conllu files for all the languages 
- `grid_scripts/eval_all.sh` evaluates the decoded files using the official conllu 2018 shared task script 

