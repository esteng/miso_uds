"""
A :class:`~miso.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~miso.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from miso.data.dataset_readers.dataset_reader import DatasetReader
from miso.data.dataset_readers.penn_tree_bank import PennTreeBankConstituencySpanDatasetReader
from miso.data.dataset_readers.semantic_role_labeling import SrlReader
from miso.data.dataset_readers.semantic_dependency_parsing import SemanticDependenciesDatasetReader
from miso.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from miso.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader
from miso.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from miso.data.dataset_readers.language_modeling import LanguageModelingDatasetReader
from miso.data.dataset_readers.aida_ere import AidaEreDatasetReader
