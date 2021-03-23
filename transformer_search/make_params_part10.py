from itertools import product 
import numpy as np
np.random.seed(12) 

layers = [6]
init_scales = [ 512, 768, 1024]
nheads = [4, 8, 16]
dropout = [0.20, 0.33]
warmup = [ 8000, 10000, 12000]
ff_dim = [1024, 2048]
lr = [0.000] 

all_combos = product(init_scales, layers, nheads, dropout, ff_dim, warmup, lr)
all_combos = [x for x in all_combos]

n_variants = 40

all_ids = [i for i in range(len(all_combos))]

chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, n_variants, replace = False)]

with open("opt_part_10.txt", "w") as f1:
    for row in chosen_combos:
        row = [str(x) for x in row]
        f1.write(" ".join(row) + "\n") 


