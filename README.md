# Stog: the String-to-Graph project

This project is based on python 3.6 and the newest version of PyTorch.
Stog is a deep learning framework used in Blab that contains re-usable 
components for deep learning models.

For a detailed overview of Stog as a deep learning framework,
see [OVERVIEW.md](https://gitlab.hltcoe.jhu.edu/szhang/stog/tree/master/docs/OVERVIEW.md).
For a tutorial on how to add a new model, task and dataset in Stog, see
[TUTORIAL.md]((https://gitlab.hltcoe.jhu.edu/szhang/stog/tree/master/docs/TUTORIAL.md).

## Installation
Via conda

- `conda env create -f stog.env -n stog_env`
- `pip install -e . -r requirements.txt`


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

