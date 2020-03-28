from decomp import UDSCorpus
from decomp.semantics.uds import UDSGraph
import networkx as nx 
import json

class TestUDSCorpus(UDSCorpus):
    """
    Wrapper object to create a UDS class which can load new input strings and empy
    graphs, for use at test time with new inputs 
    """
    def __init__(self, graphs):
        super().__init__(self)
        self._graphs = graphs
    
    @classmethod
    def from_lines(cls, path):
        with open(path) as f1:
            lines = f1.readlines()

        graphs = {}
        for i, line in enumerate(lines):
            sentence = line.strip() 
            empty_graph = nx.DiGraph()
            empty_graph.add_node(f"-root-0")
            empty_graph.nodes[f"-root-0"]['type'] = 'root'
            empty_graph.nodes[f"-root-0"]['domain'] = 'semantics'
            empty_graph.nodes[f"-root-0"]['frompredpatt'] = False
            empty_graph.nodes[f"-root-0"]['sentence'] = sentence
            name = f"test_graph_{i}"
            graph_data = nx.adjacency_data(empty_graph)
            g = UDSGraph.from_dict(graph_data, name) 
            graphs[name] = g
        
        return cls(graphs) 
       
    @classmethod
    def from_single_line(cls, line):
        lines = [line]
        graphs = {}
        for i, line in enumerate(lines):
            sentence = line.strip() 
            empty_graph = nx.DiGraph()
            empty_graph.add_node(f"test-root-0")
            empty_graph.nodes[f"test-root-0"]['type'] = 'root'
            empty_graph.nodes[f"test-root-0"]['domain'] = 'semantics'
            empty_graph.nodes[f"test-root-0"]['frompredpatt'] = False
            empty_graph.nodes[f"test-root-0"]['sentence'] = sentence
            name = f"test_graph_{i}"
            graph_data = nx.adjacency_data(empty_graph)
            g = UDSGraph.from_dict(graph_data, name) 
            graphs[name] = g
        
        return cls(graphs) 
