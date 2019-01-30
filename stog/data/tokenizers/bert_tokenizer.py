from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from stog.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN


class AMRBertTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        super(AMRBertTokenizer, self).__init__(*args, **kwargs)

    @overrides
    def tokenize(self, tokens, pos_tags):
        split_tokens, no_hashtag_tokens, split_pos_tags = [], [], []
        for token, pos_tag in zip(tokens, pos_tags):
            for i, sub_token in enumerate(self.wordpiece_tokenizer.tokenize(token)):
                split_tokens.append(sub_token)
                if i == 0:
                    no_hashtag_tokens.append(sub_token)
                else:
                    assert sub_token[:2] == '##'
                    no_hashtag_tokens.append(sub_token[2:])
                split_pos_tags.append(pos_tag + '_{}'.format(i))

        split_tokens = ['[CLS]'] + split_tokens + ['[SEP]']
        no_hashtag_tokens = ['[CLS]'] + no_hashtag_tokens + ['[SEP]']
        split_pos_tags = ['[CLS]'] + split_pos_tags + ['[SEP]']
        token_ids = self.convert_tokens_to_ids(split_tokens)
        return split_tokens, token_ids, no_hashtag_tokens, split_pos_tags
