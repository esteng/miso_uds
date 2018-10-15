from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator

reader = UniversalDependenciesDatasetReader()

train_dataset = reader.read("./UD_English-EWT/en_ewt-ud-train.conllu")

vocab = Vocabulary.from_instances(train_dataset)

iterator = BasicIterator(batch_size = 2)
iterator.index_with(vocab)
for item in iterator(train_dataset):
    import pdb;pdb.set_trace()
