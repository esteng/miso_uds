import pytest
import sys 
import os 
import numpy as np 

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from miso.data.dataset_readers.ud_parsing.ud import UDGraph
from miso.data.dataset_readers.ud_syntax import UDDatasetReader

def assert_dict(produced, expected):
    for key in expected:
        assert(produced[key] == expected[key])

def test_get_list_data():
    data_path = os.path.join(path, "data", "UD", "UD_German-GSD", "de_gsd-ud-dev.conllu")
    conllu_dicts = UDDatasetReader.parse_conllu_file(data_path) 
    test_dict = conllu_dicts[0] 
    graph = UDGraph(test_dict)
    list_data = graph.get_list_data() 
    
    expected = {'syn_tokens': ['Manasse', 'ist', 'ein', 'einzigartiger', 'Parfümeur', '.'], 
                'syn_head_indices': ['5', '5', '5', '5', '0', '5'], 
                'syn_head_tags': ['nsubj', 'cop', 'det', 'amod', 'root', 'punct'], 
                'syn_node_mask': np.array([1, 1, 1, 1, 1, 1]), 
                'syn_edge_mask': np.array([1, 1, 1, 1, 1, 1]), 
                'src_tokens': ['Manasse', 'ist', 'ein', 'einzigartiger', 'Parfümeur', '.'], 
                'src_pos_tags': ['PROPN', 'AUX', 'DET', 'ADJ', 'NOUN', 'PUNCT'], 
                'src_token_ids': None, 'src_token_subword_index': None, 
                'true_conllu_dict': [{'ID': '1', 'form': 'Manasse', 'lemma': 'Manasse', 'upos': 'PROPN', 'xpos': 'NN', 'feats': 'Case=Nom|Gender=Fem|Number=Sing', 'head': '5', 'deprel': 'nsubj', 'deps': '_', 'misc': '_'}, {'ID': '2', 'form': 'ist', 'lemma': 'sein', 'upos': 'AUX', 'xpos': 'VAFIN', 'feats': 'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin', 'head': '5', 'deprel': 'cop', 'deps': '_', 'misc': '_'}, {'ID': '3', 'form': 'ein', 'lemma': 'ein', 'upos': 'DET', 'xpos': 'ART', 'feats': 'Case=Nom|Definite=Ind|Gender=Masc|Number=Sing|PronType=Art', 'head': '5', 'deprel': 'det', 'deps': '_', 'misc': '_'}, {'ID': '4', 'form': 'einzigartiger', 'lemma': 'einzigartig', 'upos': 'ADJ', 'xpos': 'ADJA', 'feats': 'Case=Nom|Gender=Masc|Number=Sing', 'head': '5', 'deprel': 'amod', 'deps': '_', 'misc': '_'}, {'ID': '5', 'form': 'Parfümeur', 'lemma': 'Parfümeur', 'upos': 'NOUN', 'xpos': 'NN', 'feats': 'Case=Nom|Gender=Masc|Number=Sing', 'head': '0', 'deprel': 'root', 'deps': '_', 'misc': 'SpaceAfter=No'}, {'ID': '6', 'form': '.', 'lemma': '.', 'upos': 'PUNCT', 'xpos': '$.', 'feats': '_', 'head': '5', 'deprel': 'punct', 'deps': '_', 'misc': '_'}]}

    assert_dict(list_data, expected) 
