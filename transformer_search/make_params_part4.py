from itertools import product 
import numpy as np
np.random.seed(12) 

init_scales = [16, 128, 512]
layers = [6, 8, 12] 
init_scales = [32, 64, 128, 256, 512]
nheads = [4, 8]
dropout = [0.20, 0.33]
warmup = [8000, 16000]
lr = [1e-5, 1e-4, 5e-3, 7.5e-3]

all_combos = product(init_scales, layers, nheads, dropout, warmup, lr)
all_combos = [x for x in all_combos]

n_variants = 40

all_ids = [i for i in range(len(all_combos))]

chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, n_variants, replace = False)]

with open("opt_part_4.txt", "w") as f1:
    for row in chosen_combos:
        row = [str(x) for x in row]
        f1.write(" ".join(row) + "\n") 


