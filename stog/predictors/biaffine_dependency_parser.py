import json
from typing import Dict, Any, List, Tuple

from overrides import overrides

from stog.utils.registrable import Registrable
from stog.utils.checks import ConfigurationError
from stog.utils.string import JsonDict, sanitize
from stog.data import DatasetReader, Instance
from stog.models import Model
from stog.utils.archival import Archive, load_archive
from stog.utils.string import START_SYMBOL, END_SYMBOL
from stog.predictors.predictor import Predictor
from stog.data.tokenizers.word_splitter import SpacyWordSplitter
from stog.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from stog.data.fields import TextField
from stog.data.tokenizers import Token

@Predictor.register('AMRParser')
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._token_indexers = dict(
            decoder_tokens=SingleIdTokenIndexer(namespace="decoder_token_ids"),
            decoder_characters=TokenCharactersIndexer(namespace="decoder_token_characters")
        )

    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        spacy_tokens = self._tokenizer.split_words(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        if self._dataset_reader.use_language_specific_pos: # type: ignore
            # fine-grained part of speech
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            # coarse-grained part of speech (Universal Depdendencies format)
            pos_tags = [token.pos_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)


    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            return sanitize(outputs) 

    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        return {"tokens" : line.split()}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField(
            [Token(START_SYMBOL)] + [Token(x) for x in json_dict["tokens"]] + [Token(END_SYMBOL)],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )
        fields["amr_tokens"] = tokens
        return Instance(fields)
    
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return outputs["predictions"] + "\n\n"
        


