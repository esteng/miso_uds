import sys
import re
import json
import argparse
import re
from collections import defaultdict

from stog.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader

def read_alignments(file_path):
    alignment = []
    with open(file_path) as f:
        for line in f:
            alignment.append(
                    {
                        int(item.split("-")[1]) : int(item.split("-")[0])
                    for item in line.strip().split() }
            )
    return alignment

def no_reorder(instance, token_with_alignment):
    return re.sub(
        "# ::save-date", "# ::reorder {}\n# ::save-date".format(json.dumps([i for i in range(len(token_with_alignment))])), 
        instance.fields['amr'].metadata.__str__()
    )

def fully_reorder(instance, token_with_alignment):

    sorted_token_by_alignment = [item[0] for item in sorted(token_with_alignment, key=lambda x: x[1])]
    
    reorder_info = [ i for i in range(len(token_with_alignment))] 

    for i, item in enumerate(sorted_token_by_alignment):
        reorder_info[item] = i


    return re.sub(
        "# ::save-date", "# ::reorder {}\n# ::save-date".format(json.dumps(reorder_info)), 
        instance.fields['amr'].metadata.__str__()
    )

def node_reorder(
        instance, 
        token_with_alignment,
        head_first=False,
        sort_by_head_tag=False
):
    graph = instance.fields["amr"].metadata.graph
    
    tgt_list = [item for item in token_with_alignment]

    def get_list_depth(graph):
        visited = defaultdict(int)
        depth_list = []

        def dfs(node, depth):

            depth_list.append(depth)
            
            if len(node.attributes) > 1:
                for _type, token in node.attributes:
                    if _type != "instance":
                        depth_list.append(depth + 1)
                

            if len(graph._G[node]) > 0 and visited[node] == 0:
                visited[node] = 1
                for child_node, child_relation in graph._G[node].items():
                    dfs(child_node, depth + 1)

        dfs(graph.variable_to_node[graph._top], 0)

        return depth_list
    
    tgt_depth = get_list_depth(graph)

    def split_nodes(depth_list):
        root_depth = depth_list[0]
        assert root_depth == min(depth_list)

        node_spans = [(0, 1)]

        if len(depth_list) < 1:
            return []

        curr_depth = root_depth + 1
        curr_start = 1
        for idx, depth in enumerate(depth_list[2:]):
            #print(curr_start, idx, depth)
            if curr_depth >= depth:
                node_spans.append((curr_start, idx + 2))
                curr_start = idx + 2
        
        if node_spans[-1][-1] != len(depth_list):
            node_spans.append((curr_start, len(depth_list)))
    
        return node_spans
    
    
    def reorder_subseq(start, end):
        
        sub_tgt_list = tgt_list[start:end]
        sub_tgt_depth = tgt_depth[start:end]

        if end - start < 2:
            return

       
        sub_spans = split_nodes(tgt_depth[start:end])

        token_align_start_indices = [
            (
                tgt_list[start + start_idx], 
                start_idx,
                end_idx - start_idx
            ) for (start_idx, end_idx) in sub_spans
        ]

        if head_first:
            if len(sub_spans) == 2:
                reorder_subseq(start + sub_spans[1][0], start + sub_spans[1][1])
                return
            head = token_align_start_indices[0]
            token_align_start_indices = token_align_start_indices[1:]

        if sort_by_head_tag:
            assert head_first
            tgt_head_tags = instance.fields["head_tags"].labels[start: end]
            list_to_sort = [((align, tgt_head_tags[start_idx]), start_idx, length) for align, start_idx, length in token_align_start_indices]
            arg_nodes = list(filter(lambda x: re.match('^ARG[\d]+(-of)*$', x[0][1]), list_to_sort))  
            opt_nodes = list(filter(lambda x: re.match('^op[\d]+$', x[0][1]), list_to_sort))  
            snt_nodes = list(filter(lambda x: re.match('^snt[\d]+$', x[0][1]), list_to_sort))  
            other_nodes = [ item for item in list_to_sort if not (item in arg_nodes or item in opt_nodes or item in snt_nodes)]
            sorted_token = other_nodes + arg_nodes + opt_nodes + snt_nodes
            
        else:
            sorted_token = sorted(token_align_start_indices, key=lambda x: x[0][1])
        
        if head_first:
            sorted_token = [head] + sorted_token

        sub_seq_start_idx = 0
        new_sub_span = []
        new_sub_tgt_list = []
        new_sub_tgt_depth = []

        for _, orig_start_idx, length in sorted_token:
            new_sub_span.append((sub_seq_start_idx, sub_seq_start_idx + length))
            new_sub_tgt_list += sub_tgt_list[orig_start_idx: orig_start_idx + length]
            new_sub_tgt_depth += sub_tgt_depth[orig_start_idx: orig_start_idx + length]
            sub_seq_start_idx += length

        
        tgt_list[start: end] = new_sub_tgt_list
        tgt_depth[start: end] = new_sub_tgt_depth

        for s, e in new_sub_span:
            reorder_subseq(s + start, e + start) 

    reorder_subseq(0, len(tgt_list))
    
    reorder_info = [ i for i in range(len(token_with_alignment))] 
    for i, item in enumerate(tgt_list):
        reorder_info[item[0]] = i

    return re.sub(
        "# ::save-date", "# ::reorder {}\n# ::save-date".format(json.dumps(reorder_info)), 
        instance.fields['amr'].metadata.__str__()
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--type", choices=["none", "source", "node", "node_head_first", "head_tags"], default="none")
    
    args = parser.parse_args()

    dataset_reader = AbstractMeaningRepresentationDatasetReader(skip_first_line=False)
    
    alignment = read_alignments(args.alignment) 
    
    for idx, instance in enumerate(dataset_reader._read(args.input)):
        token_with_alignment = []
        for i in range(len(instance.fields["tgt_tokens"].tokens[1:-1])):
            if i in alignment[idx]:
                token_with_alignment.append((i, alignment[idx][i]))
            else:
                token_with_alignment.append((i, i))
    
    
        
        if args.type == "none":
            amr_string = no_reorder(instance, token_with_alignment)
        elif args.type == "source":
            amr_string = fully_reorder(instance, token_with_alignment)
        elif args.type == "node":
            amr_string = node_reorder(instance, token_with_alignment)
        elif args.type == "node_head_first":
            amr_string = node_reorder(instance, token_with_alignment, head_first=True)
        elif args.type == "head_tags":
            amr_string = node_reorder(instance, token_with_alignment, head_first=True, sort_by_head_tag=True)
        else:
            raise NotImplementedError
            
        print(amr_string + "\n")
