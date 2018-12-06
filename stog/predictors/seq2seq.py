from overrides import overrides
import json
from stog.utils.registrable import Registrable
from stog.utils.checks import ConfigurationError
from stog.utils.string import JsonDict, sanitize
from stog.data import DatasetReader, Instance
from stog.models import Model
from stog.utils.archival import Archive, load_archive
from stog.predictors.predictor import Predictor
from stog.utils.string import START_SYMBOL, END_SYMBOL

@Predictor.register('Seq2Seq')
class Seq2SeqPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source" : source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_batch_instance(self, instances):
        outputs = []
        _outputs = super(Seq2SeqPredictor, self).predict_batch_instance(instances)
        for output in _outputs:
            pred_token_indexes = output['predictions']
            copy_indexes = output.get('copy_indexes', None)
            tokens = self._model.vocab.get_tokens_from_list(pred_token_indexes, 'decoder_token_ids')
            if END_SYMBOL in tokens:
                tokens = tokens[:tokens.index(END_SYMBOL)]
                copy_indexes = copy_indexes[:len(tokens)] if copy_indexes else None
            outputs.append(dict(
                tokens=tokens,
                copy_indexes=copy_indexes
            ))
        return outputs

    @overrides
    def dump_line(self, output):
        pred_token_index = output["predictions"]
        pred_token_str = []
        pred_coref_str = []
        for ii, index in enumerate(output["predictions"]):
            if index == self._model.vocab.get_token_index(END_SYMBOL, "decoder_token_ids"):
                break
            pred_token_str.append(
                self._model.vocab.get_token_from_index(index, "decoder_token_ids")
            )
            pred_coref_str.append(
                str(output['copy_indexes'][ii])
            )
        

        dict_to_print = {
            "tokens" : " ".join(pred_token_str),
            "coref" : " ".join(pred_coref_str)
        }
        return json.dumps(dict_to_print) + '\n'
