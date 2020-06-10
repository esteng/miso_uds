import sys
import os 
import pickle as pkl
from subprocess import Popen, PIPE
from multiprocessing import Pool, cpu_count
import argparse
from typing import List, Tuple, Dict, Iterator 
from collections import namedtuple 
from tqdm import tqdm

from decomp.semantics.predpatt import PredPattCorpus
from decomp.semantics.uds import UDSCorpus

sys.path.insert(0, "/home/hltcoe/estengel/miso/")
from miso.data.dataset_readers.decomp.decomp import DecompGraph
from miso.metrics.s_metric.s_metric import S, TEST1 
from miso.metrics.s_metric.repr import Triple
from miso.metrics.s_metric import utils
from miso.commands.s_score import compute_args, ComputeTup
# Desired functionality: compute UD Parses from text in parallel, concatenatate them, create predpatt corpus, convert to arbor_graph

global_cmd = "./execute_java.sh {input_path} {output_path}"
input_dir = "/exp/estengel/miso/baselines/inputs"
output_dir = "/exp/estengel/miso/baselines/output"

def uds_worker(tup):
    os.chdir("/home/hltcoe/estengel/miso/baseline")
    lines, line_id = tup
    line_id = str(line_id)
    input_path = os.path.join(input_dir, line_id)
    #output_path = os.path.join(output_dir, f"{line_id}.conllu")
    output_path = output_dir
    with open(input_path, "w") as f1:
        for line in lines:
            f1.write(line.strip() + "\n")

    cmd_str = global_cmd.format(input_path = input_path, output_path = output_path)
    p = Popen(cmd_str.split(), stdout = PIPE, stderr = PIPE)
    out, errs = p.communicate()

    with open(os.path.join(output_dir, "worker_outputs", f"{line_id}.err"), "w") as f1:
        f1.write(errs.decode("utf8"))
    with open(os.path.join(output_dir, "worker_outputs", f"{line_id}.out"), "w") as f1:
        f1.write(out.decode("utf8"))

def get_uds_lines(lines, n_cpus):
    chunksize = int(len(lines) / n_cpus) + 1
    print('chunksize', chunksize)
    # chunk lines
    chunks = []
    curr_chunk = []
    for i, line in enumerate(lines):
        curr_chunk.append(line)

        if len(curr_chunk) == chunksize:
            chunks.append(curr_chunk)
            curr_chunk = []
    # add last chunk
    if curr_chunk != []:
        chunks.append(curr_chunk)

    chunks = [(chunk, i) for i, chunk in enumerate(chunks)]
        
    print(f"Making {n_cpus} workers for {len(chunks)} chunks with {len(chunks[0][0])} lines per job")
    pool = Pool(n_cpus)
    res = pool.map(uds_worker, chunks)
    
    file_content = []
    for i in range(len(chunks)):
        with open(os.path.join(output_dir, f"{i}.conllu")) as f1:
            conllu_lines = f1.read()
            file_content.append((i, conllu_lines))

    # sort by 
    file_content=sorted(file_content, key = lambda x: x[0])
    #join sorted lines into one file 
    all_lines = "".join([x[1] for x in file_content]) 
    return all_lines

def get_lines_and_graphs(split):
    # circular, but we'll read this from the Decomp corpus  
    print(f"reading corpus from {split}")
    corpus = UDSCorpus(split = split)
    lines = []
    graphs = {}
    for i, (id, graph) in enumerate(corpus.items()):
        # get true graph  
        true_arbor_graph = get_arbor_graph(graph)
        # if no semantics in gold, skip everything
        if true_arbor_graph is None:
            continue

        graphs[i] = true_arbor_graph
        # get text for prediction
        lines.append(graph.sentence)

    return lines, graphs

def get_uds_corpus(uds_text, split):
    corpus_path = os.path.join(output_dir, f"{split}.conllu")
    corpus = UDSCorpus.from_conll(corpus = uds_text, name = split)
    return corpus    

def get_arbor_graph(pp_graph):
    dg = DecompGraph(pp_graph)
    __, __, arbor_graph = dg.get_list_node()
    return arbor_graph


def compute_smetric(true_graphs: List[DecompGraph],
                    pred_graphs: List[DecompGraph],
                    args: namedtuple,
                    semantics_only: bool):
    """
    compute s-score between lists of decomp graphs
    """
    print(len(true_graphs), len(pred_graphs)) 
    assert(len(true_graphs) == len(pred_graphs))

    total_match_num, total_test_num, total_gold_num = 0, 0, 0

    for g1, g2 in tqdm(zip(true_graphs, pred_graphs), total = len(true_graphs)):
        instances1, relations1, attributes1 = DecompGraph.get_triples(g1, semantics_only)
        instances1 = [Triple(x[1], x[0], x[2]) for x in instances1]
        attributes1 = [Triple(x[1], x[0], x[2]) for x in attributes1]
        relations1 = [Triple(x[1], x[0], x[2]) for x in relations1]
        try:
            instances2, relations2, attributes2 = DecompGraph.get_triples(g2, semantics_only)
        except AttributeError:
            # None predicted
            instances2, relations2, attributes2 = [], [], []
        instances2 = [Triple(x[1], x[0], x[2]) for x in instances2]
        attributes2 = [Triple(x[1], x[0], x[2]) for x in attributes2]
        relations2 = [Triple(x[1], x[0], x[2]) for x in relations2]

        best_mapping, best_match_num, test_triple_num, gold_triple_num = S.get_best_match(
                instances1, attributes1, relations1,
                instances2, attributes2, relations2, args)

        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num

    precision, recall, best_f_score = utils.compute_f(
        total_match_num, total_test_num, total_gold_num)

    return precision, recall, best_f_score


 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-split", type = str, default = "dev")
    parser.add_argument("--precomputed", action = "store_true")
    parser.add_argument("--semantics-only", action = "store_true")
    parser.add_argument("--drop-syntax", action = "store_true")
    parser.add_argument("--nodes", default = 1, type = int)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    if not args.precomputed:
        #n_cpus = cpu_count() 
        n_cpus = 8
        lines, true_graphs = get_lines_and_graphs(args.input_split)
        with open(os.path.join(output_dir, f"{args.input_split}.graphs"), "wb") as f1:
            pkl.dump(true_graphs, f1)

        uds = get_uds_lines(lines, n_cpus)
        # pickle true graphs

        with open(os.path.join(output_dir, f"{args.input_split}.conllu"), "w") as f1:
            f1.write(uds)

    else:
        with open(os.path.join(output_dir, f"{args.input_split}.conllu")) as f1:
            uds = f1.read()
        with open(os.path.join(output_dir, f"{args.input_split}.graphs"), "rb") as f1:
            true_graphs = pkl.load(f1)

    corpus = get_uds_corpus(uds, args.input_split)
    pred_graphs = {}
    for i, e in enumerate(corpus):
        pg = get_arbor_graph(corpus[e])
        pred_graphs[i] = pg

    true_graphs = [x[1] for x in sorted(true_graphs.items(), key = lambda x: x[0])]
    pred_graphs = [x[1] for x in sorted(pred_graphs.items(), key = lambda x: x[0])]

    if args.output_path is not None:
        with open(args.output_path, "wb") as f1:
            zipped = [x for x in zip(true_graphs, pred_graphs)]
            pkl.dump(zipped, f1)        

    c_args = ComputeTup(**compute_args)
    p, r, f1 = compute_smetric(true_graphs, pred_graphs, c_args, args.semantics_only)

    print(f"precision: {p}, recall: {r}, F1: {f1}")


