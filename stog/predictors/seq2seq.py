from overrides import overrides
import json
from stog.utils.registrable import Registrable
from stog.utils.checks import ConfigurationError
from stog.utils.string import JsonDict, sanitize
from stog.data import DatasetReader, Instance
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
        gen_vocab_size = self._model.vocab.get_vocab_size('decoder_token_ids')
        _outputs = super(Seq2SeqPredictor, self).predict_batch_instance(instances)
        for instance, output in zip(instances, _outputs):
            copy_vocab = instance.fields['src_copy_vocab'].metadata
            pred_token_indexes = output['predictions']
            coref_indexes = output.get('coref_indexes', None)
            tokens = []
            copy_indicators = []
            for token_index in pred_token_indexes:
                if token_index >= gen_vocab_size:
                    copy_token_index = token_index - gen_vocab_size
                    tokens.append(copy_vocab.get_token_from_idx(copy_token_index))
                    copy_indicators.append(1)
                else:
                    tokens.append(self._model.vocab.get_token_from_index(
                        token_index, 'decoder_token_ids'
                    ))
                    copy_indicators.append(0)

            if END_SYMBOL in tokens:
                tokens = tokens[:tokens.index(END_SYMBOL)]
                coref_indexes = coref_indexes[:len(tokens)] if coref_indexes else None
            outputs.append(dict(
                tokens=tokens,
                coref_indexes=coref_indexes,
                copy_indicators=copy_indicators
            ))
        return outputs

    @overrides
    def dump_line(self, output):
        return ' '.join(output['tokens']) + '\n'
        # return ' '.join('{}_{}_{}'.format(x, y, z) for x, y, z in zip(output['tokens'], output['coref_indexes'], output['copy_indicators'])) + '\n'
        dict_to_print = {
            "tokens": " ".join(output["tokens"]),
            "corefs": output["coref_indexes"],
        }
        return json.dumps(dict_to_print) + '\n'
