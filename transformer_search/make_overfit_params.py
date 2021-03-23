from itertools import product 
import numpy as np
np.random.seed(12) 

init_scales = [2, 4, 8, 16, 32, 64, 128, 256, 512]
layers = [i for i in range(2,12)]
nheads = [2, 4, 8, 16]
dropout = [0.00]
warmup = [1, 5, 10, 500, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 4000, 8000, 16000]
lr = [1e-5, 1e-4, 5e-3, 7.5e-3, 1e-3, 1e-2]

all_combos = product(init_scales, layers, nheads, dropout, warmup, lr)
all_combos = [x for x in all_combos]

n_variants = 1000

all_ids = [i for i in range(len(all_combos))]

chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, n_variants, replace = False)]

with open("overfit_opt.txt", "w") as f1:
    for row in chosen_combos:
        row = [str(x) for x in row]
        f1.write(" ".join(row) + "\n") 


