import pytest
import sys 
import os 

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from decomp import UDSCorpus
from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph 
from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax 

def assert_dict(produced, expected):
    for key in expected:
        assert(produced[key] == expected[key])

@pytest.fixture
def load_dev_graphs():
    all_dev_graphs = UDSCorpus(split="dev") 
    test_graphs = {"basic": all_dev_graphs["ewt-dev-1"], 
                   "coref": all_dev_graphs["ewt-dev-2"],
                   "long": all_dev_graphs["ewt-dev-19"]}

    return test_graphs

def test_get_list_data_concat_after(load_dev_graphs):
    # test concat-after 
    d_graph = DecompGraphWithSyntax(load_dev_graphs['basic'], syntactic_method="concat-after")
    list_data = d_graph.get_list_data(bos="@start@", eos="@end@", max_tgt_length = 100) 
    expected = {"tgt_tokens": ["@start@", "@@ROOT@@", "comes", "AP", "the", "story", "this", "From",  '@syntax-sep@', 'comes', 'AP', 'story', ':', 'From', 'the', 'this', '@end@'],
                "head_tags": ['dependency', 'dependency', 'dependency', 'EMPTY', 'dependency', 'EMPTY', 'EMPTY', 'SEP', 'root', 'nmod', 'nsubj', 'punct', 'case', 'det', 'det'],
                "head_indices": [0, 1, 2, 3, 2, 5, 2, -1, 0, 9, 9, 9, 10, 10, 11]}

    assert_dict(list_data, expected) 

def test_get_list_data_concat_before(load_dev_graphs):
    # test concat-after 
    d_graph = DecompGraphWithSyntax(load_dev_graphs['basic'], syntactic_method="concat-before")
    list_data = d_graph.get_list_data(bos="@start@", eos="@end@", max_tgt_length = 100) 
    expected = {"tgt_tokens": ['@start@', 'comes', 'AP', 'story', ':', 'From', 'the', 'this', '@syntax-sep@', '@@ROOT@@', 'comes', 'AP', 'the', 'story', 'this', 'From', '@end@'],
                "head_indices": [0, 1, 1, 1, 2, 2, 3, -1, 0, 9, 10, 11, 10, 13, 10],
                "head_tags": ['root', 'nmod', 'nsubj', 'punct', 'case', 'det', 'det', 'SEP', 'dependency', 'dependency', 'dependency', 'EMPTY', 'dependency', 'EMPTY', 'EMPTY']} 

    assert_dict(list_data, expected) 

def test_get_list_data_syntax_basic(load_dev_graphs): 
    # test 1: basic  
    d_graph = DecompGraph(load_dev_graphs["basic"]) 
    list_data = d_graph.get_list_data(bos="@start@", eos="@end@")
    expected = {"tgt_tokens": ["@start@", "@@ROOT@@", "comes", "AP", "the", "story", "this", "From", "@end@"],
                "tgt_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "tgt_attributes": [{}, {}, {"factuality-factual": {"value": 0.2626969516277313, "confidence": 1.0}, "time-dur-centuries": {"value": -1.1705, "confidence": 1.0}, "time-dur-days": {"value": 0.8576, "confidence": 1.0}, "time-dur-decades": {"value": -1.1145, "confidence": 1.0}, "time-dur-forever": {"value": -1.4625, "confidence": 1.0}, "time-dur-hours": {"value": 0.9937, "confidence": 1.0}, "time-dur-instant": {"value": -1.4102, "confidence": 1.0}, "time-dur-minutes": {"value": -0.9328, "confidence": 1.0}, "time-dur-months": {"value": -1.2142, "confidence": 1.0}, "time-dur-seconds": {"value": 0.8932, "confidence": 1.0}, "time-dur-weeks": {"value": -1.3236, "confidence": 1.0}, "time-dur-years": {"value": 0.9253, "confidence": 1.0}, "genericity-pred-dynamic": {"value": -0.2368, "confidence": 1.0}, "genericity-pred-hypothetical": {"value": -0.3223, "confidence": 1.0}, "genericity-pred-particular": {"value": 1.9132, "confidence": 1.0}}, {"genericity-arg-abstract": {"value": -1.6835, "confidence": 1.0}, "genericity-arg-kind": {"value": -1.6835, "confidence": 1.0}, "genericity-arg-particular": {"value": 2.5166, "confidence": 1.0}}, {}, {"wordsense-supersense-noun.Tops": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.act": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.animal": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.artifact": {"value": -0.5789, "confidence": 1.0}, "wordsense-supersense-noun.attribute": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.body": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.cognition": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.communication": {"value": 1.5722, "confidence": 1.0}, "wordsense-supersense-noun.event": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.feeling": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.food": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.group": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.location": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.motive": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.object": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.person": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.phenomenon": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.plant": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.possession": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.process": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.quantity": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.relation": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.shape": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.state": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.substance": {"value": -2.101, "confidence": 1.0}, "wordsense-supersense-noun.time": {"value": -2.101, "confidence": 1.0}, "genericity-arg-abstract": {"value": -1.3927, "confidence": 1.0}, "genericity-arg-kind": {"value": -0.1493, "confidence": 1.0}, "genericity-arg-particular": {"value": 2.5166, "confidence": 1.0}}, {}, {}, {}],
            "tgt_copy_indices": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "tgt_tokens_to_generate": ["@start@", "@@ROOT@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@end@"]
            }

    assert_dict(list_data, expected) 

def test_get_list_data_syntax_reentrancy(load_dev_graphs): 
    # test 2: corefferent/reentrant node 
    d_graph = DecompGraph(load_dev_graphs["coref"])
    list_data = d_graph.get_list_data(bos="@start@", eos="@end@")
    expected = {"tgt_tokens": ["@start@", "@@ROOT@@", "nominated", "Bush", 
                                "President", "Tuesday", "individuals", "two", 
                                "on", "replace", "Bush", "President", "jurists", 
                                "retiring", "on", "federal", "courts", "in", "the", 
                                "Washington", "area", "to", "@end@"], 
                "tgt_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                "tgt_tokens_to_generate": ["@start@", "@@ROOT@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@end@"],
                "head_tags": ["dependency", "dependency", "dependency", "EMPTY", "dependency", "dependency", "EMPTY", 
                              "EMPTY", "dependency", "dependency", "EMPTY", "dependency", "EMPTY", "EMPTY", "EMPTY", 
                              "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"], 
                "head_indices": [0, 1, 2, 3, 2, 2, 6, 2, 1, 9, 10, 9, 12, 12, 12, 12, 12, 12, 12, 12, 9],
                "node_name_list": ["@start@", "dummy-semantics-root", "ewt-dev-2-semantics-pred-5", "ewt-dev-2-semantics-arg-2", 
                                  "ewt-dev-2-syntax-1", "ewt-dev-2-semantics-arg-4", "ewt-dev-2-semantics-arg-7", 
                                  "ewt-dev-2-syntax-6", "ewt-dev-2-syntax-3", "ewt-dev-2-semantics-pred-9", 
                                  "ewt-dev-2-semantics-arg-2", "ewt-dev-2-syntax-1", "ewt-dev-2-semantics-arg-11", 
                                  "ewt-dev-2-syntax-10", "ewt-dev-2-syntax-12", "ewt-dev-2-syntax-13", 
                                  "ewt-dev-2-syntax-14", "ewt-dev-2-syntax-15", "ewt-dev-2-syntax-16", 
                                  "ewt-dev-2-syntax-17", "ewt-dev-2-syntax-18", "ewt-dev-2-syntax-8", "@end@"]
                }

    assert_dict(list_data, expected) 

#def test_get_list_data_syntax_long(load_dev_graphs): 
#    # test 3: long data
#    d_graph = DecompGraph(load_dev_graphs["long"])
#    list_data = d_graph.get_list_data(bos="@start@", eos="@end@")
#    
#    print(list_data) 
#    assert(2==1) 

