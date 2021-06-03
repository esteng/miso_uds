from typing import List
import pdb 

import networkx as nx 

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

    def tgt_str_to_list(self, tgt_str):
        """
        convert lispress to list for MISO 
        """
        # special characters: (,), ,#
        split_str = tgt_str.split(" ")


        def tgt_str_to_program_helper(split_span: List[str], parent_idx, argn = "arg0"): 
            # base case: no further function 
            if "(" not in split_span and ")" not in split_span:
                curr_idx = self.node_idx_list[-1] + 1

                name_list = split_span
                node_idxs = [curr_idx + i for i in range(len(split_span))]
                edge_heads = [parent_idx] + [curr_idx for i in range(len(split_span)-1)]
                edge_types = [argn] + [f"arg{i}" for i in range(len(split_span)-1)]

                self.node_name_list += name_list
                self.node_idx_list += node_idxs
                self.edge_type_list += edge_types
                self.edge_head_list += edge_heads
                pdb.set_trace() 
            else:
                try:
                    curr_idx = self.node_idx_list[-1] + 1
                except IndexError:
                    curr_idx = 0
                name_list = []
                node_idxs = []
                edge_heads = []
                edge_types = []
                if parent_idx == -1:
                    edge_heads.append(0)
                    edge_types.append('root')
                else:
                    edge_heads.append(parent_idx)
                    edge_types.append(argn)

                name_list.append(split_span[0])
                node_idxs.append(curr_idx)

                bp = 1

                # add the front part of the program, e.g. program name and any singleton args 
                while split_span[bp] != "(":
                    name_list.append(split_span[bp])
                    node_idxs.append(curr_idx + bp)
                    edge_heads.append(curr_idx)
                    edge_types.append(f"arg{bp-1}")
                    
                    bp+=1

                pdb.set_trace() 
                self.node_name_list += name_list
                self.node_idx_list += node_idxs
                self.edge_type_list += edge_types
                self.edge_head_list += edge_heads

                # recurse on rest of program 
                tgt_str_to_program_helper(get_matching_span(split_span[bp:]), parent_idx=curr_idx, argn = f"arg{bp}")

        def get_matching_span(text_left_to_right):
            num_parens = 1
            to_ret = []
            for tok in text_left_to_right:
                to_ret.append(tok)
                if tok == "(":
                    num_parens += 1
                if tok == ")":
                    num_parens -= 1
                if num_parens == 0:
                    break
            # break off parens 
            return to_ret[1:-1]

        pdb.set_trace() 
        
        tgt_str_to_program_helper(get_matching_span(split_str[0:]), parent_idx=-1)

    def tgt_str_to_graph(self, tgt_str):
        """
        convert lispress to a graph AST 
        """
        graph = nx.DiGraph()
        # special characters: (,), ,#
        split_str = tgt_str.split(" ")

        def tgt_str_to_program_helper(split_span: List[str], parent_idx) -> nx.DiGraph: 
            # base case: no further function 
            if "(" not in split_span and ")" not in split_span:
                return graph

            else:
                curr_idx = parent_idx + 1
                graph.add_node(curr_idx, text= split_span[0])
                if parent_idx == -1:
                    graph.add_edge(curr_idx, curr_idx, type="root")
                else:
                    graph.add_edge(curr_idx, parent_idx, type='arg0')
                bp = 1
                while split_span[bp] != "(":
                    graph.add_node(curr_idx + bp, text = split_span[bp])
                    graph.add_edge(curr_idx + bp, curr_idx, type=f"arg{bp}")
                    bp+=1

                return tgt_str_to_program_helper(get_matching_span(split_span[bp:]), parent_idx=curr_idx)

        def get_matching_span(text_left_to_right):
            num_parens = 1
            to_ret = []
            for tok in text_left_to_right:
                to_ret.append(tok)
                if tok == "(":
                    num_parens += 1
                if tok == ")":
                    num_parens -= 1
                if num_parens == 0:
                    break
            # break off parens 
            return to_ret[1:-1]

        pdb.set_trace() 
        graph = tgt_str_to_program_helper(get_matching_span(split_str[1:]), parent_idx=-1)
        return graph  

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