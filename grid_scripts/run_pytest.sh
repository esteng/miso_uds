#!/bin/bash
#$ -j yes
#$ -N pytest 
#$ -l 'mem_free=50G,h_rt=24:00:00'
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_uds/grid_logs/pytest.out 

cd /home/hltcoe/estengel/miso_uds
source ~/envs/miso_res/bin/activate

pytest 
