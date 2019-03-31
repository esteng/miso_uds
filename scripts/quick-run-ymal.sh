#!/bin/bash
source activate stog
work_dir=$PWD

date | tee $work_dir/run.log
echo Work path: $work_dir | tee -a $work_dir/run.log
sed -e "s~\s*serialization_dir:\s&\w*\s.*~  serialization_dir: \&serialization_dir $PWD/ckpt~g" $1 > $work_dir/exp-run.yaml

code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

cd $code_dir
echo Code path: $PWD | tee -a $work_dir/run.log

git status | grep "nothing to commit" > /dev/null

if [ $? == 1 ]
then
  echo
  echo Please run the code before you commit everything! | tee -a $work_dir/run.log
  echo
  git status
  exit 1
fi

echo Branch name `git branch | grep \* | cut -d ' ' -f2` | tee -a $work_dir/run.log
echo Current commit | tee -a $work_dir/run.log
git rev-parse HEAD | tee -a $work_dir/run.log
git log -1 --pretty=%B | head -n 1 | tee -a $work_dir/run.log

CUDA_VISIBLE_DEVICES=`free-gpu` python -m stog.commands.train $work_dir/exp-run.yaml
