"""
A :class:`~miso.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~miso.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
#from miso.data.dataset_readers.ccgbank import CcgBankDatasetReader
#from miso.data.dataset_readers.conll2003 import Conll2003DatasetReader
#from miso.data.dataset_readers.conll2000 import Conll2000DatasetReader
#from miso.data.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
#from miso.data.dataset_readers.coreference_resolution import ConllCorefReader, WinobiasReader
from miso.data.dataset_readers.dataset_reader import DatasetReader
#from miso.data.dataset_readers.event2mind import Event2MindDatasetReader
#from miso.data.dataset_readers.language_modeling import LanguageModelingReader
#from miso.data.dataset_readers.multiprocess_dataset_reader import MultiprocessDatasetReader
from miso.data.dataset_readers.penn_tree_bank import PennTreeBankConstituencySpanDatasetReader
#from miso.data.dataset_readers.reading_comprehension import SquadReader, TriviaQaReader, QuACReader
from miso.data.dataset_readers.semantic_role_labeling import SrlReader
from miso.data.dataset_readers.semantic_dependency_parsing import SemanticDependenciesDatasetReader
from miso.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
#from miso.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
#from miso.data.dataset_readers.snli import SnliReader
from miso.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from miso.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader
#from miso.data.dataset_readers.stanford_sentiment_tree_bank import (
#        StanfordSentimentTreeBankDatasetReader)
#from miso.data.dataset_readers.quora_paraphrase import QuoraParaphraseDatasetReader
#from miso.data.dataset_readers.semantic_parsing import (
#        WikiTablesDatasetReader, AtisDatasetReader, NlvrDatasetReader, TemplateText2SqlDatasetReader)
#from miso.data.dataset_readers.semantic_parsing.quarel import QuarelDatasetReader
