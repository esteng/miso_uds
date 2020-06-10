import sys
import os 
import pickle as pkl
from subprocess import Popen, PIPE
from multiprocessing import Pool, cpu_count
import argparse
from typing import List, Tuple, Dict, Iterator 
from collections import namedtuple 
from tqdm import tqdm
import tempfile 
import numpy as np 

from decomp.semantics.predpatt import PredPattCorpus
from decomp.semantics.uds import UDSCorpus

file_path = os.path.abspath(__file__)
baseline_path = os.path.dirname(file_path) 
miso_path = os.path.join(os.path.dirname(baseline_path))
sys.path.insert(0, miso_path) 

from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax
from miso.commands.conllu_score import ConlluScorer, compute_args, ComputeTup
from miso.metrics.conllu import evaluate_wrapper, UDError

# Desired functionality: compute UD Parses from text in parallel, concatenatate them, create predpatt corpus, convert to arbor_graph

global_cmd = "./execute_java.sh {input_path} {output_path}"
#input_dir = "/exp/estengel/miso/baselines/inputs"
#output_dir = "/exp/estengel/miso/baselines/output"

input_dir = "/Users/Elias/miso_research/baseline/inputs"
output_dir = "/Users/Elias/miso_research/baseline/output"


def uds_worker(tup):
    os.chdir(baseline_path) 
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
        true_graph = get_decomp_graph(graph)
        # if no semantics in gold, skip everything
        if true_graph is None:
            continue

        graphs[i] = true_graph
        # get text for prediction
        lines.append(graph.sentence)

    return lines, graphs

def get_uds_corpus(uds_text, split):
    corpus_path = os.path.join(output_dir, f"{split}.conllu")
    corpus = UDSCorpus.from_conll(corpus = uds_text, name = split)
    return corpus    

def get_decomp_graph(pp_graph):
    dg = DecompGraphWithSyntax(pp_graph)
    return dg

def compute_conllu_score(true_graphs, pred_graphs):
    las_scores, mlas_scores, blex_scores = [], [], []

    for true_graph, pred_graph in zip(true_graphs, pred_graphs):
        true_graph.syntactic_method = "encoder-side"
        pred_graph.syntactic_method = "encoder-side"
        true_conllu_dict = true_graph.get_list_data()['true_conllu_dict']
        pred_conllu_dict = pred_graph.get_list_data()['true_conllu_dict']
        true_conllu_str = ConlluScorer.conllu_dict_to_str(true_conllu_dict) 
        pred_conllu_str = ConlluScorer.conllu_dict_to_str(pred_conllu_dict)  
    
    # make temp files
        with tempfile.NamedTemporaryFile("w") as true_file, \
            tempfile.NamedTemporaryFile("w") as pred_file:

            true_file.write(true_conllu_str)
            pred_file.write(pred_conllu_str) 
            true_file.seek(0) 
            pred_file.seek(0) 
            compute_args["gold_file"] = true_file.name
            compute_args["system_file"] = pred_file.name

            args = ComputeTup(**compute_args)
            try:
                score = evaluate_wrapper(args)
                las_scores.append(100 * score["LAS"].f1)
                mlas_scores.append(100 * score["MLAS"].f1)
                blex_scores.append(100 * score["BLEX"].f1)
            except UDError:
                las_scores.append(0)
                mlas_scores.append(0)
                blex_scores.append(0)

    return np.mean(las_scores), np.mean(mlas_scores), np.mean(blex_scores)  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-split", type = str, default = "dev")
    parser.add_argument("--precomputed", action = "store_true")
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--line-limit", type=int, default = -1) 

    args = parser.parse_args()

    if not args.precomputed:
        n_cpus = cpu_count() - 2
        #n_cpus = 8
        lines, true_graphs = get_lines_and_graphs(args.input_split)
        if args.line_limit > 0:  
            lines = lines[0:args.line_limit]
            true_graphs = {i:v for i, v in true_graphs.items() 
                            if i < args.line_limit}

        uds = get_uds_lines(lines, n_cpus)
        
        # save predictions
        with open(os.path.join(output_dir, f"{args.input_split}.conllu"), "w") as f1:
            f1.write(uds)

    else:
        lines, true_graphs = get_lines_and_graphs(args.input_split)

        with open(os.path.join(output_dir, f"{args.input_split}.conllu")) as f1:
            uds = f1.read()

    corpus = get_uds_corpus(uds, args.input_split)
    pred_graphs = {}
    for i, e in enumerate(corpus):
        pg = get_decomp_graph(corpus[e])
        pred_graphs[i] = pg

    true_graphs = [x[1] for x in sorted(true_graphs.items(), key = lambda x: x[0])]
    pred_graphs = [x[1] for x in sorted(pred_graphs.items(), key = lambda x: x[0])]

    #if args.output_path is not None:
    #    with open(args.output_path, "wb") as f1:
    #        zipped = [x for x in zip(true_graphs, pred_graphs)]
    #        pkl.dump(zipped, f1)        

    las, mlas, blex = compute_conllu_score(true_graphs, pred_graphs)  

    print(f"LAS: {las}, MLAS: {mlas}, BLEX: {blex}") 

