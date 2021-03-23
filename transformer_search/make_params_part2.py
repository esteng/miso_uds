from itertools import product 
import numpy as np
np.random.seed(12) 


init_scales = [128, 256, 512]
layers = [12] 
nheads = [8]
dropout = [0.1, 0.20]
warmup = [10000, 12000, 14000]
decay = [3e-9, 3e-8, 3e-7]

all_combos = product(init_scales, layers, nheads, dropout, warmup, decay)
all_combos = [x for x in all_combos]

n_variants = 38

all_ids = [i for i in range(len(all_combos))]

chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, n_variants, replace = False)]


with open("opt_part_2.txt", "w") as f1:
    for row in chosen_combos:
        row = [str(x) for x in row]
        f1.write(" ".join(row) + "\n") 


