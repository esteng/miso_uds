from itertools import product 
import numpy as np
np.random.seed(12) 


init_scales = [32, 64, 128, 256, 512]
layers = [8, 10, 12, 14] 
nheads = [4, 8, 16]
dropout = [0.20, 0.33, 0.50]
warmup = [1000, 2000, 4000, 8000]

all_combos = product(init_scales, layers, nheads, dropout, warmup) 
all_combos = [x for x in all_combos]

n_variants = 38

all_ids = [i for i in range(len(all_combos))]

chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, n_variants, replace = False)]


with open("combos.txt", "w") as f1:
    for row in chosen_combos:
        row = [str(x) for x in row]
        f1.write(" ".join(row) + "\n") 


