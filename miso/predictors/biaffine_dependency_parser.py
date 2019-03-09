import json
from typing import Dict, Any, List, Tuple

from overrides import overrides

from miso.utils.registrable import Registrable
from miso.utils.checks import ConfigurationError
from miso.utils.string import JsonDict, sanitize
from miso.data import DatasetReader, Instance
from miso.utils.string import START_SYMBOL, END_SYMBOL
from miso.predictors.predictor import Predictor
from miso.data.tokenizers.word_splitter import SpacyWordSplitter
from miso.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from miso.data.fields import TextField, SequenceLabelField
from miso.data.tokenizers import Token

import json
@Predictor.register('BiaffineParser')
class BiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.
    """
    def __init__(self, model, dataset_reader: DatasetReader) -> None:
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
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            return sanitize(outputs)

    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        return json.loads(line.strip())

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField(
            [Token(START_SYMBOL)] + [Token(x) for x in json_dict["tokens"].split(' ')] + [Token(END_SYMBOL)],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )
        fields["amr_tokens"] = tokens
        coref_int = json_dict['coref']
        fields["coref_index"] = SequenceLabelField(
            labels=[0] + [x + 1 for x in coref_int] + [len(coref_int)],
            sequence_field=fields["amr_tokens"],
            label_namespace="coref_tags",
        )
        return Instance(fields)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return outputs["predictions"] + "\n\n"



