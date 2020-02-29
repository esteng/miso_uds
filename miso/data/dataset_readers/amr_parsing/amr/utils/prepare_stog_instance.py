import re
from collections import Counter, defaultdict

from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from miso.data.dataset_readers.amr_parsing.amr.src_copy_vocab import SourceCopyVocabulary


def is_abstract_token(token):
    return re.search(r'^([A-Z]+_)+\d+$', token) or re.search(r'^\d0*$', token)


def is_english_punct(c):
    return re.search(r'^[,.?!:;"\'-(){}\[\]]$', c)


def find_similar_token(token, tokens):
    token = re.sub(r'-\d\d$', '', token) # .lower())
    for i, t in enumerate(tokens):
        if token == t:
            return tokens[i]
        # t = t.lower()
        # if (token == t or
        #     (t.startswith(token) and len(token) > 3) or
        #     token + 'd' == t or
        #     token + 'ed' == t or
        #     re.sub('ly$', 'le', t) == token or
        #     re.sub('tive$', 'te', t) == token or
        #     re.sub('tion$', 'te', t) == token or
        #     re.sub('ied$', 'y', t) == token or
        #     re.sub('ly$', '', t) == token
        # ):
        #     return tokens[i]
    return None


def trim_very_long_tgt_tokens(tgt_tokens, head_tags, head_indices, node_to_idx, max_tgt_length):
    tgt_tokens = tgt_tokens[:max_tgt_length]
    head_tags = head_tags[:max_tgt_length]
    head_indices = head_indices[:max_tgt_length]
    for node, indices in node_to_idx.items():
        invalid_indices = [index for index in indices if index >= max_tgt_length]
        for index in invalid_indices:
            indices.remove(index)
    return tgt_tokens, head_tags, head_indices, node_to_idx


def add_source_side_tags_to_target_side(src_tokens, src_tags, tgt_tokens):
    assert len(src_tags) == len(src_tokens)
    tag_counter = defaultdict(lambda: defaultdict(int))
    for src_token, src_tag in zip(src_tokens, src_tags):
        tag_counter[src_token][src_tag] += 1

    tag_lut = {DEFAULT_OOV_TOKEN: DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN: DEFAULT_OOV_TOKEN}
    for src_token in set(src_tokens):
        tag = max(tag_counter[src_token].keys(), key=lambda x: tag_counter[src_token][x])
        tag_lut[src_token] = tag

    tgt_tags = []
    for tgt_token in tgt_tokens:
        sim_token = find_similar_token(tgt_token, src_tokens)
        if sim_token is not None:
            index = src_tokens.index(sim_token)
            tag = src_tags[index]
        else:
            tag = DEFAULT_OOV_TOKEN
        tgt_tags.append(tag)

    return tgt_tags, tag_lut


def prepare_stog_instance(
        src_tokens, src_pos_tags, tgt_tokens, head_tags, head_indices, node_to_idx,
        bos=None, eos=None, bert_tokenizer=None, max_tgt_length=None):

    if max_tgt_length is not None:
        tgt_tokens, head_tags, head_indices, node_to_idx = trim_very_long_tgt_tokens(
            tgt_tokens, head_tags, head_indices, node_to_idx, max_tgt_length)

    copy_offset = 0
    if bos:
        tgt_tokens = [bos] + tgt_tokens
        copy_offset += 1
    if eos:
        tgt_tokens = tgt_tokens + [eos]

    # Target side Coreference
    tgt_indices = [i for i in range(len(tgt_tokens))]

    for node, indices in node_to_idx.items():
        if len(indices) > 1:
            copy_idx = indices[0] + copy_offset
            for token_idx in indices[1:]:
                tgt_indices[token_idx + copy_offset] = copy_idx

    tgt_copy_map = [(token_idx, copy_idx) for token_idx, copy_idx in enumerate(tgt_indices)]
    tgt_copy_indices = tgt_indices[:]

    for i, copy_index in enumerate(tgt_copy_indices):
        # Set the coreferred target to 0 if no coref is available.
        if i == copy_index:
            tgt_copy_indices[i] = 0

    tgt_token_counter = Counter(tgt_tokens)
    tgt_copy_mask = [0] * len(tgt_tokens)
    for i, token in enumerate(tgt_tokens):
        if tgt_token_counter[token] > 1:
            tgt_copy_mask[i] = 1

    # Source Copy
    src_token_ids = None
    src_token_subword_index = None
    src_copy_vocab = SourceCopyVocabulary(src_tokens)
    src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
    src_copy_map = src_copy_vocab.get_copy_map(src_tokens)
    tgt_pos_tags, pos_tag_lut = add_source_side_tags_to_target_side(src_tokens, src_pos_tags, tgt_tokens)

    if bert_tokenizer is not None:
        src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)

    src_anonym_indicators = [1 if is_abstract_token(t) else 0 for t in src_tokens]
    src_copy_invalid_ids = set(src_copy_vocab.index_sequence(
        [t for t in src_tokens if is_english_punct(t)]))

    return {
        "tgt_tokens": tgt_tokens,
        "tgt_indices": tgt_indices,
        "tgt_pos_tags": tgt_pos_tags,
        "tgt_copy_indices": tgt_copy_indices,
        "tgt_copy_map": tgt_copy_map,
        "tgt_copy_mask": tgt_copy_mask,
        "src_tokens": src_tokens,
        "src_token_ids": src_token_ids,
        "src_token_subword_index": src_token_subword_index,
        "src_anonym_indicators": src_anonym_indicators,
        "src_pos_tags": src_pos_tags,
        "src_copy_vocab": src_copy_vocab,
        "src_copy_indices": src_copy_indices,
        "src_copy_map": src_copy_map,
        "pos_tag_lut": pos_tag_lut,
        "head_tags": head_tags,
        "head_indices": head_indices,
        "src_copy_invalid_ids": src_copy_invalid_ids
    }
