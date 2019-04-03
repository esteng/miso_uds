#!/bin/bash
set -e
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path_to_LDC2016T10 [output_dir]"
  echo "Default output_dir is current directory ($PWD)"
  exit 1
fi

LDC2016T10=$1
if [ -z "$2" ]
then
  OUTPUT=$PWD
else
  OUTPUT=$2
fi

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# semeval 2014
# English Only, use section 20 as dev set as recommaned in README.txt
mkdir -p $OUTPUT/2014
for format in dm pas pcedt
do
  python $SCRIPTPATH/sdp_split_train_dev.py --input $LDC2016T10/data/2014/$format.sdp --output-dir $OUTPUT/2014 --dev-section 20
  cp $LDC2016T10/data/2014/test/$format.tt $OUTPUT/2014/test.$format.sdp
  cp $LDC2016T10/data/2014/test/$format.sdp $OUTPUT/2014/gold.$format.sdp
done

# semeval 2015
# English, Chinese, Czech
# All use section 20 as dev set (English and Czech are recommaned in this way, while no preference for Chinese)
# There are also in domain and out of domain test in semeval 2015
mkdir -p $OUTPUT/2015
# English: available in all there formats 
lang=en
for format in dm pas psd
do
  python $SCRIPTPATH/sdp_split_train_dev.py --input $LDC2016T10/data/2015/$lang.$format.sdp --output-dir $OUTPUT/2015 --dev-section 20 --skip-first-line
  cp $LDC2016T10/data/2015/test/$lang.id.$format.tt $OUTPUT/2015/test.id.$format.sdp
  cp $LDC2016T10/data/2015/test/$lang.id.$format.sdp $OUTPUT/2015/gold.id.$format.sdp
  cp $LDC2016T10/data/2015/test/$lang.ood.$format.tt $OUTPUT/2015/test.ood.$format.sdp
  cp $LDC2016T10/data/2015/test/$lang.ood.$format.sdp $OUTPUT/2015/gold.ood.$format.sdp
done

# Chinese: only available in the pas format, no out of domain test
format=pas 
lang=cz
python $SCRIPTPATH/sdp_split_train_dev.py --input $LDC2016T10/data/2015/$lang.$format.sdp --output-dir $OUTPUT/2015 --dev-section 20 --skip-first-line
cp $LDC2016T10/data/2015/test/$lang.id.$format.tt $OUTPUT/2015/test.id.$format.sdp
cp $LDC2016T10/data/2015/test/$lang.id.$format.sdp $OUTPUT/2015/gold.id.$format.sdp

# Czech: only available in the psd format
format=psd
lang=cs
python $SCRIPTPATH/sdp_split_train_dev.py --input $LDC2016T10/data/2015/$lang.$format.sdp --output-dir $OUTPUT/2015 --dev-section 20 --skip-first-line
cp $LDC2016T10/data/2015/test/$lang.id.$format.tt $OUTPUT/2015/test.id.$format.sdp
cp $LDC2016T10/data/2015/test/$lang.id.$format.sdp $OUTPUT/2015/gold.id.$format.sdp
cp $LDC2016T10/data/2015/test/$lang.ood.$format.tt $OUTPUT/2015/test.ood.$format.sdp
cp $LDC2016T10/data/2015/test/$lang.ood.$format.sdp $OUTPUT/2015/gold.ood.$format.sdp

# CCD
# English Only.
mkdir -p $OUTPUT/ccd
python $SCRIPTPATH/sdp_split_train_dev.py --input $LDC2016T10/data/ccd/train.sdp --output-dir $OUTPUT/ccd --dev-section 20 --skip-first-line
cp $LDC2016T10/data/ccd/test.sdp $OUTPUT/ccd/test.sdp


