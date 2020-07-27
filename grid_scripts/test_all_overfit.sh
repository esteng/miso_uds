#!/bin/bash
#$ -j yes
#$ -N decomp_pytest
#$ -l 'mem_free=5G,h_rt=24:00:00'
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/pytest_all.out


cd /home/hltcoe/estengel/miso_research/

source ~/envs/miso_res/bin/activate

OUT_DIR="/home/hltcoe/estengel/miso_research/test_logs/"

#pytest test/test_decomp_data.py &> ${OUT_DIR}/data.out 
pytest test/test_interface_overfit.py &> ${OUT_DIR}/interface.out
#pytest test/test_transformer_overfit.py &> ${OUT_DIR}/transformer.out
#pytest test/test_intermediate_overfit.py &> ${OUT_DIR}/intermediate.out
#pytest test/test_xlmr.py &> ${OUT_DIR}/xlmr.out
#pytest test/test_commands.py &> ${OUT_DIR}/commands.out
