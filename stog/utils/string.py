
def namespace_match(pattern: str, namespace: str):
    """
    Adopted from AllenNLP:
        https://github.com/allenai/allennlp/blob/v0.6.1/allennlp/common/util.py#L164

    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == '*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False