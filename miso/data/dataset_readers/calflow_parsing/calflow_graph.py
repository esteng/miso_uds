from typing import List, Any
import pdb 

import networkx as nx
from networkx.readwrite.multiline_adjlist import parse_multiline_adjlist 

from task_oriented_dialogue_as_dataflow_synthesis.src.dataflow.core.program import Program, Expression

#from task_oriented_dialogue_as_dataflow_synthesis.src.dataflow.core.dialogue import Turn

class CalFlowGraph:
    def __init__(self, 
                src_str: str,
                tgt_str: str):
        #self.program = self.tgt_str_to_program(tgt_str)
        self.src_str = src_str
        self.tgt_str = tgt_str
        self.node_name_list = []
        self.node_idx_list  = []
        self.edge_type_list = []
        self.edge_head_list  = []
        self.parent = -1 
        self.argn = 0

        self.ast = self.get_ast(self.tgt_str.split(" "))
        self.preorder_ast_traversal(self.ast, parent = -1, is_function = True, argn=-1) 

    def get_ast(self, split_str: List[str]) -> List[Any]:
        """
        Turn a lispress into a nested list which is the AST 
        borrowed from: https://gist.github.com/roberthoenig/30f08b64b6ba6186a2cdee19502040b4
        """
        ast = []
        i = 0
        while i < len(split_str): 
            curr_symb = split_str[i]
            if curr_symb == "(":
                inner_list = []
                # find closing bracket
                n_parens = 1
                while n_parens != 0:
                    i += 1
                    if i >= len(split_str):
                        raise ValueError("Invalid lispress expression: unmatched open paren")
                    curr_symb = split_str[i]
                    if curr_symb == "(":
                        n_parens += 1
                    elif curr_symb == ")":
                        n_parens -= 1
                    if n_parens != 0:
                        inner_list.append(curr_symb)
                ast.append(self.get_ast(inner_list))
            elif curr_symb == ")":
                raise ValueError("Invalid lispress expression: unmatched close paren")
            else:
                ast.append(curr_symb)

            i += 1
        return ast 

    def preorder_ast_traversal(self, 
                               ast: List[Any], 
                               parent: int = -1, 
                               is_function: bool = False, 
                               has_children: bool = False,
                               argn: int = 0):
        def is_atom(piece: Any) -> bool:
            if type(piece) == list:
                return False
            return True

        # base case: we've reached a terminal element 
        if len(self.node_idx_list) > 0:
            curr_idx = self.node_idx_list[-1] + 1
        else:
            curr_idx = 0
        if is_atom(ast):
            self.node_idx_list.append(curr_idx)
            self.node_name_list.append(ast)
            self.edge_head_list.append(parent)
            self.edge_type_list.append(argn-1)
            # if it's a function, update parent and reset arg counter
            if is_function and has_children: 
                self.argn = 0

        # recurse 
        else:
            for i, inner_ast in enumerate(ast):

                is_function = (i == 0)
                has_children = len(inner_ast) > 1 and len(ast) > 1
                if is_function and is_atom(inner_ast[0]) and has_children:
                    new_parent = curr_idx
                else:
                    new_parent = parent 
                self.preorder_ast_traversal(inner_ast, parent=parent, is_function=is_function, has_children = has_children, argn = self.argn)    
                self.argn += 1 
                parent = new_parent 

    def lists_to_ast(self, node_name_list: List[str], edge_head_list: List[int], edge_type_list: List[int]) -> List[Any]:
        """
        convert predicted lists back to an AST 
        """
        full_ast = []
        curr_ast = []
        is_function = [False for i in range(len(node_name_list))]
        for i, edge_type in enumerate(edge_type_list):
            if edge_type == 0:
                # parent is a function if current edge is an arg0
                parent_idx = edge_head_list[i]
                is_function[parent_idx] = True

        # use digraph to store data and then convert 
        graph = nx.DiGraph() 
        for i, (node_name, edge_head, edge_type, is_func) in enumerate(zip(node_name_list, edge_head_list, edge_type_list, is_function)):
            graph.add_node(i, node_name = node_name, is_func = is_func)
            # root self-edges
            if edge_head < 0:
                edge_head = 0
            graph.add_edge(edge_head, i, type=edge_type)

        return graph 

    def digraph_to_lispress(self, graph: nx.DiGraph, root_id: int = 0, lispress = ""): 
        def get_children(node):
            # exclude the self-loop to avoid infinite recursion 
            children = [e[1] for e in graph.edges if e[0] == node and e[1] != node]
            return children 
        
        def get_parent(node):
            parents = [e[0] for e in graph.edges if e[1] == node and e[0] != node]
            assert(len(parents) == 1)
            return parents[0]

        children = get_children(root_id)
        if len(children) > 0:
            lispress = f"( {graph.nodes[root_id]['node_name']} " + " ".join([f"{self.digraph_to_lispress(graph, child, lispress)}" for child in children]) + " )"
            return lispress
        else:
            # deal with singleton functions 
            node_name = graph.nodes[root_id]['node_name']

            # check for multiple upper case  
            uc_count = sum([1 if node_name[i].isupper() else 0 for i in range(len(node_name))])
            # catches everything except "Now", "Today", etc. 
            is_func = False
            if uc_count > 1 and not node_name.isupper(): 
                is_func = True
            else: 
                # check parent
                parent = get_parent(root_id)
                if graph.nodes[parent]['node_name'].strip() == "?=" and node_name != "#":
                    is_func = True

            if is_func: 
                node_name = f"( {node_name} )"
            return node_name


    def lists_to_program(self, node_name_list: List[str], edge_head_list: List[int], edge_type_list: List[int]) -> Program:
        """
        convert predicted lists back to a Program
        """
        pass  

    def pred_ast_to_lispress(self, pred_ast: List[Any]) -> str: 
        """
        convert a predicted AST to a lispress 
        """
        pass 

    def get_list_data(self, 
                     bos: str = None, 
                     eos: str = None, 
                     bert_tokenizer = None, 
                     max_tgt_length: int = None):
        """
        Converts SMCalFlow graph into a linearized list of tokens, indices, edges, and affiliated metadata 

        """


        return {
            "tgt_tokens" : tgt_tokens,
            "tgt_indices": tgt_indices,
            "tgt_copy_indices" : tgt_copy_indices,
            "tgt_copy_map" : tgt_copy_map,
            "tgt_tokens_to_generate": tgt_tokens_to_generate, 
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "tgt_copy_mask" : tgt_copy_mask,
            "src_tokens" : src_tokens,
            "src_token_ids" : src_token_ids,
            "src_token_subword_index" : src_token_subword_index,
            "src_must_copy_tags" : src_must_copy_tags,
            "src_copy_vocab" : src_copy_vocab,
            "src_copy_indices" : src_copy_indices,
            "src_copy_map" : src_copy_map,
            "pos_tag_lut": pos_tag_lut,
            "src_copy_invalid_ids" : src_copy_invalid_ids,
            "arbor_graph": arbor_graph,
            "node_name_list": node_name_list,
        }

    