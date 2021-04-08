# Readme for running UD experiments

## current UD/UDS experiments 

## Current Multilingual experiments 
The setup here is different, since there is no parallel data here. Instead, we train a model on one task and use it to initialize a model for another task.
So for example, for German UD parsing, we would
1. train a UD parsing model on English (EWT) 
2. train a joint UD and UDS model on English (EWT)
3. train a German UD parsing model initialized from model (1) 
4. train a German UD model initialized from (2) 
5. compare the performance between models (3) and (4) 

In the other direction (pretraining on UD, evaluating on UDS) we follow these steps: 

we would training a French UD parsing model from scratch, and comparing that with a French UD parsing model where the compatible weights are initialized by weights learned on English UDS parsing. The settings here are 
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

