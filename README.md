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


## Configurations 
MISO model instantiation happens via a config file, which is stored as a `.jsonnet` file. 


## Models 

### Joint syntax-semantics models 

### Encoders 

### Contextualized encoders 

## Training 

## Testing 


