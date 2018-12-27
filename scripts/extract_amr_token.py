import sys
from stog.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader

dataset_reader = AbstractMeaningRepresentationDatasetReader(skip_first_line=False)

for instance in dataset_reader._read(sys.argv[1]):
    amr_tokens = ' '.join([x.text for x in instance.fields["tgt_tokens"].tokens[1:-1]])
    print(amr_tokens)
