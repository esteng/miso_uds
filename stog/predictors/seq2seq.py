from overrides import overrides

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
    def dump_line(self, output):
        pred_token_index = output["predictions"]
        pred_token_str = []
        for index in output["predictions"]:
            if index == self._model.vocab.get_token_index(END_SYMBOL, "decoder_token_ids"):
                break
            pred_token_str.append(
                self._model.vocab.get_token_from_index(index, "decoder_token_ids")
            )
        
        #TODO: print attention
        return " ".join(pred_token_str)  + "\n"
