#!/bin/bash
#$ -N baseline
#$ -j yes
#$ -o /home/hltcoe/estengel/miso/baseline/logs/test_run_baseline_syntax
#$ -l 'mem_free=10G,h_rt=36:00:00'
#$ -cwd 



source activate miso
cd /home/hltcoe/estengel/miso/baseline
echo "RUNNING ON COMMIT:"
git reflog | head -n 1

python -u parse_text.py --input-split test --drop-syntax --nodes=8 --output-path /exp/estengel/miso/decomp/outputs/baseline_syntax_graphs.pkl
