import sys
import re

from stog.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader

dataset_reader = AbstractMeaningRepresentationDatasetReader(skip_first_line=False)

for instance in dataset_reader._read(sys.argv[1]):
    src_tokens = ' '.join([x.text for x in instance.fields["src_tokens"].tokens])
    tgt_tokens = ' '.join([x.text for x in instance.fields["tgt_tokens"].tokens[1:-1]])
    #amr_string = re.sub("\n +", " ", instance.fields['amr'].metadata.graph.__str__())
    print(src_tokens + " ||| " + tgt_tokens)
