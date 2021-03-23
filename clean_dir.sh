#!/bin/bash

cd /home/hltcoe/estengel/miso_research

for dir in ${1}/*; do
    echo "${dir}"
    ./clean_export.sh ${dir}
done
