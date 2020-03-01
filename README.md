# MISO: Multimodal In(puts), Semantic Out(puts)

- MISO is a deep learning framework used in Blab that contains re-usable 
components for deep learning models.
- MISO is heavily based on [AllenNLP](https://github.com/allenai/allennlp).

## Codebase Overview

<table>
<tr>
    <td><b> data </b></td>
    <td> The directory to place the dataset you want to work with. </td>
</tr>
<tr>
    <td><b> docs </b></td>
    <td> A collection of documents. </td>
</tr>
<tr>
    <td><b> experiments </b></td>
    <td> The directory to place scripts to run your experiments. </td>
</tr>
<tr>
    <td><b> miso </b></td>
    <td> The core MISO code. See [README](miso/README.md) for detail. </td>
</tr>
<tr>
    <td><b> scripts </b></td>
    <td> Other scripts. </td>
</tr>
</table>

## Installation

Via conda on Linux:

- `conda create --name miso python=3.6`
- `conda activate miso`
- `pip install -r requirements.txt`

Via conda on macOS:

- `conda create --name miso python=3.6`
- `conda activate miso`
- `conda install jsonnet=0.10`
- `pip install -r requirements.txt`

## Contributing

Adding a new model to MISO is basically the same as adding a new AllenNLP model,
so getting familar with AllenNLP would be very helpful. 
I recommend you first going over 
[the AllenNLP tutorials](https://github.com/allenai/allennlp/tree/master/tutorials)
before moving forward.

## Running a Simple Language Model Example

#### Training

Example of training (on clsp grid):

```
CUDA_VISIBLE_DEVICES=`free-gpu` python -u -m miso.commands.train params/lm.yaml
```

#### Update pre-trained model

To recover (reload) and continue training from an existing model; or to override any
parameters, the yaml files can be chained by comma, e.g.

```
CUDA_VISIBLE_DEVICES=`free-gpu` python -u -m miso.commands.train params/lm.yaml,params/recover.yaml
```


