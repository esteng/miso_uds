from itertools import product 
import numpy as np
np.random.seed(12) 

layers = [6]
init_scales = [512, 768]
nheads = [8]
dropout = [0.20]
warmup = [7000, 9000]
lr = [0.000] 

all_combos = product(init_scales, layers, nheads, dropout, warmup, lr)
all_combos = [x for x in all_combos]

n_variants = 4

all_ids = [i for i in range(len(all_combos))]

chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, n_variants, replace = False)]

with open("opt_part_8.txt", "w") as f1:
    for row in chosen_combos:
        row = [str(x) for x in row]
        f1.write(" ".join(row) + "\n") 


