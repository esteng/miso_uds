# Instructions

This contains detailed list of instructions on how to add a new model,
task, or dataset reader to Stog

### Stog components
Here we describe the three main components/archictecture of the Stog framework:


- **Modules** - implementations of useful deep learning components. These
are things like [encoders](https://gitlab.hltcoe.jhu.edu/szhang/stog/tree/master/stog/modules/seq2vec_encoders),
[decoders](https://gitlab.hltcoe.jhu.edu/szhang/stog/tree/master/stog/modules/decoders), 
[attention layers](https://gitlab.hltcoe.jhu.edu/szhang/stog/tree/master/stog/modules/attention_layers),
etc.

- **Models** - Task specific compositions of modules. Some of the existing models are: 
  - [Stog](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/models/stog.py) - Model used in Sheng's AMR parsing submission
  - [LM](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/models/language_model.py) - Simple language model implemented by [Patrick](https://gitlab.hltcoe.jhu.edu/paxia)
  - [Seq2Seq](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/models/seq2seq.py)
  - *Your new model goes [here](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/models/)*

- **Data** - dataset specific IO implementations:
  - **DatasetReader** - extensions of the baseclass [DatasetReader](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/data/dataset_readers/dataset_reader.py#L26-105) 
    - *put your DatasetReader in `/stog/data/dataset_readers`*
  - **DataWriter** - extensions of the baseclass [DataWriter](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/data/data_writers/data_writer.py#L2) that writes out model output
    - *put your DataWriter in `/stog/data/data_writers`*
  - [`dataset_builder.py`](https://gitlab.hltcoe.jhu.edu/szhang/stog/blob/master/stog/data/dataset_builder.py) - group of methods for IO of data
    - *after creating your own DatasetReader and DataWriter* 

### Running experiments

Experiments in stog require a configuration file where parameters are specified.
Stog currently supports configuration files written in JSON or YAML.

Configuration files are broken into the following required subsections: