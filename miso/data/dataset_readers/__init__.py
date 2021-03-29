"""
A :class:`~miso.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~miso.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from .decomp import DecompDatasetReader
