# MISO: Multimodal In(puts), Semantic Out(puts)

- MISO is a deep learning framework used in Blab that contains re-usable 
components for deep learning models.
- MISO is heavily based on [AllenNLP](https://github.com/allenai/allennlp).
- MISO currently requires Python 3.6, PyTorch 1.4.0, and AllenNLP 0.9.0. 
Check out [requirements.txt](requirements.txt) for more prerequisites.

## Quick Links
- [Codebase Overview](#codebase-overview)
- [Installation](#installation)
- [Contributing](#contributing)
  * [The `DatasetReader`](#the--datasetreader-)
  * [Defining a `Model`](#defining-a--model-)
  * [Setting up the `Trainer`](#setting-up-the--trainer-)
  * [Making Predictions](#making-predictions)
- [Configuration](#configuration)
- [Running Existing Models](#running-existing-models)


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
    <td> The core MISO code. </td>
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
so it would be very helpful to first go over 
[the AllenNLP tutorials](https://github.com/allenai/allennlp/tree/master/tutorials)
before you proceed.

Typically to add a new model, you'll have to implement four classes:

1. `DatasetReader`, which contains the logic for reading a file of 
    data and producing a stream of `Instance`s (more about those shortly).

2. `Model`, which is a PyTorch `Module` that takes `Tensor` inputs and produces 
    a dict of outputs (including the training loss you want to optimize).

3. [Optional] `Trainer`, which handles most of the details of training models.

4. [Optional] `Predictor`, which takes inputs, converts them to `Instance`s,
    feeds them through your model, and returns results.

I recommend you diving into their base classes in AllenNLP, 
because the code itself is the best documentation. A
lso, you'll learn the best practices of developing deep learning models
by reading them. 

Below I summarize the key points of each class:

### The `DatasetReader`

In AllenNLP each training example is represented as an `Instance` consisting of 
`Field`s of various types.
A `DatasetReader` contains the logic to generate those instances (typically) 
from data stored on disk.

Typically to create a `DatasetReader` you'd implement two methods:

1. `text_to_instance` takes the inputs corresponding to a training example
   (in this case the `tokens` of the sentence and the corresponding part-of-speech `tags`),
   instantiates the corresponding `Field`s
   (in this case a `TextField` for the sentence and a `SequenceLabelField` for its tags),
   and returns the `Instance` containing those fields.

2. `_read` takes the path to an input file and returns an `Iterator` of `Instance`s.
   (It will probably delegate most of its work to `text_to_instance`.)

Check out [the AMR Dataset Reader](miso/data/dataset_readers/amr.py) in MISO
for an example. More documentation can be found in 
[AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/tagger/README.md#the-datasetreader).

More examples: [AllenNLP dataset readers](https://github.com/allenai/allennlp/tree/master/allennlp/data/dataset_readers)

### Defining a `Model`

A `Model` is a subclass of `torch.nn.Module` with a forward method that takes 
some input tensors and produces a dict of output tensors. 
How this all works is largely up to you -- the only requirement is that your 
output dict contain a `"loss"` tensor, as that's what our training code uses to 
optimize your model parameters.

Check out [the AMR Parser](miso/models/amr_parser.py) in MISO
for an example. More documentation can be found in 
[AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/tagger/README.md#defining-a-model).

More examples: [AllenNLP models](https://github.com/allenai/allennlp/tree/master/allennlp/models)

### Setting up the `Trainer`

Most of the time you don't have to implent the `Trainer`, because AllenNLP 
already includes a very full-featured `Trainer` that handles most of the 
gory details of training models. All you have to do is to set it up in the
configuration file (we'll talk about it shortly).

In case that the default trainer doesn't satisfy your needs, check out 
[the AMR Parsing Trainer](miso/training/amr_parsing_trainer.py) for an example
of how to override functions. 
Reading [the default Trainer](https://github.com/allenai/allennlp/blob/master/allennlp/training/trainer.py)
as well as [the training command](https://github.com/allenai/allennlp/blob/master/allennlp/commands/train.py)
will also help you know how to implement your own `Trainer`.

### Making Predictions

Often your model only needs to output `"loss"` and a dict of tensors during
training. During prediction, you'd need to converts the model output to
the acceptable format (e.g. a sequence of tags in the part-of-speech tagging 
task, or a dependency parse tree in the dependency parsing task).

AllenNLP provides a `Predictor` abstraction that handles the basics like 
taking inputs, converting them to `Instance`s, feeding them through your model, 
and returning JSON-serializable results. 
Please read [the base class](https://github.com/allenai/allennlp/blob/master/allennlp/predictors/predictor.py)
before implementing your own `Predictor`.

Check out [the AMR Parsing Predictor](miso/predictors/amr_parsing_predictor.py) in MISO
for an example. 

More examples: [AllenNLP predictors](https://github.com/allenai/allennlp/tree/master/allennlp/predictors)

## Configuration

Most AllenNLP or MISO objects are constructible from JSON-like parameter objects.
The configuration files are written in [Jsonnet](https://jsonnet.org/), 
which is a superset of JSON with some nice features around variable substitution.

If you want to use the same config file approach to the classes you implement,
you need to register it with a type by providing a decorator like this:
```python
@DatasetReader.register("amr")
class AMRDatasetReader:
    ...
```
Once this code has been run, AllenNLP knows that a dataset reader config with 
type "amr" refers to this class. Similarly, you need to decorate your model:
```python
@DatasetReader.register("amr_parser")
class AMRParser:
    ...
```

Now the remainder of the configuration is specified in 
[a jsonnet file](miso/training_config/transductive_semantic_parsing.jsonnet). 
For the most part it should be pretty straightforward; 
one novel piece is that Jsonnet allows us to use local variables, 
which means we can specify experimental parameters all in one place.

## Running Existing Models

- [AMR Parser]()


