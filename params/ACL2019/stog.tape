task link_data
  < in=$data
  > out
  {
    ln -s $in $out
  }

task make_yaml
  < in=$yaml
  < train=$out@link_data[Section:Train]
  < dev=$out@link_data[Section:Dev]
  < test=$out@link_data[Section:Test]
  > out
  {
    cp $in .
    file_name=`basename $in`

    data_dir=`dirname $train`
    sed -i -e "s@__DATA_PATH__@\/@g" $file_name

    train_base_name=`basename $train`
    dev_base_name=`basename $dev`
    test_base_name=`basename $test`

    sed -i -e "s@__TRAIN_FILE__@$train@g" $file_name
    sed -i -e "s@__DEV_FILE__@$dev@g" $file_name
    sed -i -e "s@__TEST_FILE__@$test@g" $file_name
    
    ln -s $file_name $out

  }

task train
  < in=$out@make_yaml
  > out
  :: num_gpu=2
  #:: .submitter=$submitter
  #:: .resource_flags=$gpu_resource_flags 
  #:: .action_flags=$action_flags
  {
    source activate stog
    ln -s $in stog.yaml
    CUDA_VISIBLE_DEVICES=`free-gpu $num_gpu` python -m stog.commands.train stog.yaml
    ln -s stog-ckpt $out
  }

task predict
  < in=$out@link_data[Section:Test]
  < model=$out@train
  > out
  :: beam_size=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'hostname=c*,gpu=1'"
  :: .action_flags=""
  {
    source activate stog
    CUDA_VISIBLE_DEVICES=`free-gpu` python -u -m stog.commands.predict \
      --archive-file $model \
      --weights-file $model/best.th \
      --input-file $in \
      --batch-size 32 --use-dataset-reader --output-file prediction.txt --silent --cuda-device 0 --beam-size $beam_size
      ln -s prediction.txt $out
  }

task postprocess
  < in=$out@predict
  > out
  {
    source activate stog
    ln -s $in pred
    /home/xma/projects/stog/experiments/amr-753/postprocess.sh pred
    ln -s pred*.expand $out
  }

task eval
  < in=$out@postprocess
  < ref=$out@link_data[Type:Origin,Section:Test]
  > out
  {
    ln -s $in input
    grep -E "tgt_ref" input | cut -d " " -f3- > tgt_ref
    grep -E "tgt_pred" input | cut -d " " -f3- > tgt_pred
    exp_dir=$PWD
    /home/xma/tools/mosesdecoder/scripts/generic/multi-bleu.perl tgt_ref < tgt_pred | tee log
    cd /home/xma/projects/stog/tools/amr-evaluation-tool-enhanced
    ./evaluation.sh $ref $exp_dir/input |tee -a log
    ln -s log $out
  }
  

plan run {
  reach eval via (Dataset: LDC2017T10) * (Type: Preprocess) * (Section: *) * (ExpName: *) * (BeamSize: *)
}

plan run_coverage {
  reach eval via (Dataset: LDC2017T10) * (Type: Preprocess) * (Section: *) * (ExpName: WO_Coverage) * (BeamSize: *)
}

plan run_glove {
  reach eval via (Dataset: LDC2017T10) * (Type: Preprocess) * (Section: *) * (ExpName: WO_Glove) * (BeamSize: *)
}

plan run_cnn {
  reach eval via (Dataset: LDC2017T10) * (Type: Preprocess) * (Section: *) * (ExpName: WO_Char_Cnn) * (BeamSize: *)
}

plan run_train_origin {
  reach eval via (Dataset: LDC2017T10) * (Type: Origin) * (Section: *) * (ExpName: Train_Origin) * (BeamSize: *)
}

plan run_src_emb {
  reach eval via (Dataset: LDC2017T10) * (Type: Preprocess) * (Section: *) * (ExpName: WO_Source_Embedding) * (BeamSize: *)
}

submitter sge :: action_flags
              :: COMMANDS
              :: TASK REALIZATION TASK_VARIABLES CONFIGURATION {
  action run {
    wrapper="ducttape_job.sh"
    echo "#$ $resource_flags" >> $wrapper
    echo "#$ $action_flags" >> $wrapper
    echo "#$ -o localhost:$PWD/qsub.stdout" >> $wrapper
    echo "#$ -e localhost:$PWD/qsub.stderr" >> $wrapper
    echo "#$ -V" >> $wrapper
    echo "#$ -S /bin/zsh" >> $wrapper
    echo "#$ -N $CONFIGURATION-$TASK-$REALIZATION" >> $wrapper
    echo "set -e # stop on errors" >> $wrapper

    echo "$TASK_VARIABLES" | grep -v "resource_flags" | grep -v "action_flags" >> $wrapper
    echo "cd $PWD" >> $wrapper
    echo "$COMMANDS" >> $wrapper

    # Use SGE's -sync option to prevent qsub from immediately returning
    qsub -sync y $wrapper
  }
}

versioner git :: repo ref {
  action checkout > dir {
    git clone $repo $dir
  }
  action repo_version > version {
    git ls-remote $repo $ref | cut -f1 > $version
  }
  # Used to confirm version after checkout
  action local_version > version date {
    git rev-parse HEAD > $version
    git log -1 | awk '/^Date/{$1=""; print}' > $date
  }
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}
