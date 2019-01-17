# Stog: the String-to-Graph project

This project is based on python 3.6 and the newest version of PyTorch.

## Baselines

- RNN Seq2Seq Model (see [this](https://gitlab.hltcoe.jhu.edu/research/mt-ie/tree/copy) for details) 
- (Optional) Transition-based Semantic Parser
 

## Installation
Via conda

- `conda env create -f stog.env -n stog_env`
- `pip install -e . -r requirements.txt`

## Instructions

For a detailed list of Instructions on how to add a model
for a new task and dataset, see [Instructions.md](https://gitlab.hltcoe.jhu.edu/szhang/stog/tree/master/docs/INSTRUCTIONS.md)

## Running a Simple Language Model Example

#### Training
Example of training (on clsp grid):

```
CUDA_VISIBLE_DEVICES=`free-gpu` python -u -m stog.commands.train params/lm.yaml
```

#### Update pre-trained model
To recover (reload) and continue training from an existing model; or to override any
parameters, the yaml files can be chained by comma, e.g.

```
CUDA_VISIBLE_DEVICES=`free-gpu` python -u -m stog.commands.train params/lm.yaml,params/recover.yaml
```

#### Testing a model

*TODO*

