#!/bin/bash
#$ -N syn_baseline
#$ -j yes
#$ -o /home/hltcoe/estengel/miso_research/baseline/logs/run_baseline_conllu
#$ -l 'mem_free=10G,h_rt=36:00:00'
#$ -cwd 

source /home/hltcoe/estengel/envs/miso_res/bin/activate
cd /home/hltcoe/estengel/miso_research/baseline
echo "RUNNING ON COMMIT:"
git reflog | head -n 1

python -u parse_syntax.py --input-split dev --output-path /exp/estengel/miso_res/interface/baseline/outputs/baseline_syntax_graphs.pkl --precomputed
