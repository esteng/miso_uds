#!/bin/bash
#$ -j yes
#$ -N visualize 
#$ -l 'mem_free=50G,h_rt=24:00:00,cpu=1'
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_research/transformer_search/logs/vis.out

source ~/envs/miso_res/bin/activate
echo "$(which jupyter)"
jupyter notebook --no-browser --port 8888

