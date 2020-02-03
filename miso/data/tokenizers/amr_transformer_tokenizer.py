from typing import List, Dict
from overrides import overrides

import numpy as np
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


@PretrainedTransformerTokenizer.register("pretrained_transformer_for_amr")
class AMRTransformerTokenizer(PretrainedTransformerTokenizer):

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        assert "bert" in model_name, "Only support BERT models."
        super(AMRTransformerTokenizer, self).__init__(model_name, *args, **kwargs)

    @property
    def unk_idx(self) -> int:
        return self._tokenizer.unk_idx

    @overrides
    def tokenize(self, tokens: List[str], split: bool = False) -> Dict:
        tokens = self._start_tokens + tokens + self._end_tokens
        if not split:
            sub_tokens = [t if t in self._tokenizer.vocab else self._tokenizer.unk_token for t in tokens]
            token_recovery_matrix = None
        else:
            sub_tokens, token_recovery_indices = [], []
            for token in tokens:
                indices = []
                for i, sub_token in enumerate(self._tokenizer.wordpiece_tokenizer.tokenize(token)):
                    indices.append(len(sub_tokens))
                    sub_tokens.append(sub_token)
                token_recovery_indices.append(indices)

            token_recovery_indices = token_recovery_indices[1:-1]  # Exclude start and end tokens.
            max_index_list_len = max(len(indices) for indices in token_recovery_indices)
            token_recovery_matrix = np.zeros((len(token_recovery_indices), max_index_list_len))
            for i, indices in enumerate(token_recovery_indices):
                for j, index in enumerate(indices):
                    token_recovery_matrix[i, j] = index

        token_ids = np.array(self._tokenizer.convert_tokens_to_ids(sub_tokens))
        return {"token_ids": token_ids, "token_recovery_matrix": token_recovery_matrix}
