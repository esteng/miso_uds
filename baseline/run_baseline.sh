#!/bin/bash
#$ -N baseline
#$ -j yes
#$ -o /home/hltcoe/estengel/miso/baseline/logs/run_baseline_replicate
#$ -l 'mem_free=10G,h_rt=36:00:00'
#$ -cwd 

source activate miso
cd /home/hltcoe/estengel/miso/baseline
python -u parse_text.py --input-split dev --semantics-only --drop-syntax --nodes=8 --output-path /exp/estengel/miso/decomp/outputs/baseline_sem_only_graphs.pkl
