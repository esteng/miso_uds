import sys
import re
import json
import argparse
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

def node_reorder(instance, token_with_alignment):
    graph = instance.fields["amr"].metadata.graph
    
    tgt_list = [item for item in token_with_alignment]

    def get_list_depth(graph):
        visited = defaultdict(int)
        depth_list = []

        def dfs(node, depth):

            depth_list.append(depth)

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
        
    
    def reorder_subseq(tgt_list, tgt_depth):
        assert len(tgt_list) == len(tgt_depth)

        if len(tgt_depth) < 2:
            return

        spans = split_nodes(tgt_depth)
        
        if len(spans) <= 1:
            return
        
        spans_with_head_alignment = [
            (tgt_list[i][-1], spans[i]) for i in len(tgt_list)
        ]

        
    split = split_nodes(tgt_depth)
    print(tgt_depth)
    for span in split:
        print(tgt_depth[span[0]: span[1]])
       
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--type", choices=["none", "full", "node"], default="none")
    
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
        else:
            amr_string = node_reorder(instance, token_with_alignment)
            
        print(amr_string + "\n")
