# Instructions

This contains detailed list of instructions on how to add a new model,
task, or dataset reader to Stog

### Stog components
Here we describe the three main components/archictecture of the Stog framework:


- **Modules** - implementations of useful deep learning components:
  - [encoders]()
  - [decoders]()
  - [embedders]()
  - [attention]() 
  
- **Models** - Task specific compositions of modules (this is where you would put your task-specific model that is composed of the smaller modules)
  - [Stog]() - Model used in Sheng's AMR parsing submission
  - [LM]() - Simple language model implemented by [Patrick]()
  - *Your new model goes here*

- **Data** - dataset specific IO implementations:
  - **DatasetReader** - extensions of the baseclass [DatasetReader]() **put link to class (line 26)**
    - *put your DatasetReader in `/stog/data/dataset_readers`*
  - **DataWriter** - extensions of the baseclass [DataWriter] that writes out model output
    - *put your DataWriter in `/stog/data/data_writers`*
  - `dataset_builder.py` - group of methods for IO of data
    - *after creating your own DatasetReader and DataWriter* 

### Running experiments

Experiments in stog require a configuration file where parameters are specified.
Stog currently supports configuration files written in JSON or YAML.

Configuration files are broken into the following required subsections: