import re


def clean(amr):
    join_model_name(amr)
    join_al_names(amr)
    split_pro_prefix(amr)
    split_anti_prefix(amr)
    replace_NT_dollar_abbr(amr)


def join_model_name(amr):
    # Joint the words starting with a cap letter which is followed by '^-\d+$'
    while True:
        span = None
        if len(amr.tokens) < 2:
            break
        for i in range(len(amr.tokens) - 1):
            x, y = amr.tokens[i: i + 2]
            if x.isupper() and re.search(r'^-\d+$', y):
                span = list(range(i, i + 2))
                joined_tokens = ''.join([x, y])
                break
        else:
            break
        amr.replace_span(span, [joined_tokens], ['NNP'], ['ENTITY'])


def join_al_names(amr):
    # Join the words starting with ['Al', '-'] or ['al', '-'].
    while True:
        span = None
        if len(amr.tokens) < 3:
            break
        for i in range(len(amr.tokens) - 2):
            x, y, z = amr.tokens[i:i + 3]
            if x in ('al', 'Al') and y == '-' and z[0].isupper():
                span = list(range(i, i + 3))
                joined_tokens = ''.join(['Al', y, z])
                break
        else:
            break
        amr.replace_span(span, [joined_tokens], ['NNP'], ['PERSON'])


def split_pro_prefix(amr):
    # Split word with 'pro-' prefix.
    while True:
        index = None
        for i, lemma in enumerate(amr.lemmas):
            if lemma.lower().startswith('pro-'):
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        _, lemma = amr.lemmas[index].split('-', 1)
        amr.replace_span([index], ['pro', lemma], ['JJ', pos], ['O', ner])


def split_anti_prefix(amr):
    # Split word with 'anti-' prefix.
    while True:
        index = None
        for i, lemma in enumerate(amr.lemmas):
            if lemma.lower().startswith('anti-'):
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        _, lemma = amr.lemmas[index].split('-', 1)
        amr.replace_span([index], ['anti', lemma], ['JJ', pos], ['O', ner])


def replace_NT_dollar_abbr(amr):
    # Replace 'NT' in front of '$' with 'Taiwan'.
    for i, token in enumerate(amr.tokens):
        if token == 'NT' and len(amr.tokens) > i + 1 and amr.tokens[i + 1] in ('$', 'dollars', 'dollar'):
            amr.replace_span([i], ['Taiwan'], ['NNP'], ['COUNTRY'])