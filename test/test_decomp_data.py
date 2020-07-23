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
                "tgt_attributes": [{}, {}, {'factuality-factual': {'confidence': 1.0, 'value': 0.967}, 'time-dur-weeks': {'confidence': 0.2564, 'value': -1.3247}, 'time-dur-decades': {'confidence': 0.2564, 'value': -1.1146}, 'time-dur-days': {'confidence': 0.2564, 'value': 0.8558}, 'time-dur-hours': {'confidence': 0.2564, 'value': 0.9952}, 'time-dur-seconds': {'confidence': 0.2564, 'value': 0.8931}, 'time-dur-forever': {'confidence': 0.2564, 'value': -1.4626}, 'time-dur-centuries': {'confidence': 0.2564, 'value': -1.1688}, 'time-dur-instant': {'confidence': 0.2564, 'value': -1.4106}, 'time-dur-years': {'confidence': 0.2564, 'value': 0.9252}, 'time-dur-minutes': {'confidence': 0.2564, 'value': -0.9337}, 'time-dur-months': {'confidence': 0.2564, 'value': -1.2142}, 'genericity-pred-dynamic': {'confidence': 0.627, 'value': -0.0469}, 'genericity-pred-hypothetical': {'confidence': 0.5067, 'value': -0.0416}, 'genericity-pred-particular': {'confidence': 1.0, 'value': 1.1753}}, {'genericity-arg-kind': {'confidence': 1.0, 'value': -1.1642}, 'genericity-arg-abstract': {'confidence': 1.0, 'value': -1.1642}, 'genericity-arg-particular': {'confidence': 1.0, 'value': 1.2257}}, {}, {'wordsense-supersense-noun.object': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.Tops': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.quantity': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.feeling': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.food': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.shape': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.event': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.motive': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.substance': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.time': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.person': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.process': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.attribute': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.artifact': {'confidence': 1.0, 'value': -1.3996}, 'wordsense-supersense-noun.group': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.animal': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.location': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.plant': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.possession': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.relation': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.phenomenon': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.cognition': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.act': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.state': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.communication': {'confidence': 1.0, 'value': 0.2016}, 'wordsense-supersense-noun.body': {'confidence': 1.0, 'value': -3.0}, 'genericity-arg-kind': {'confidence': 0.7138, 'value': -0.035}, 'genericity-arg-abstract': {'confidence': 1.0, 'value': -1.1685}, 'genericity-arg-particular': {'confidence': 1.0, 'value': 1.2257}}, {}, {}, {}],
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

