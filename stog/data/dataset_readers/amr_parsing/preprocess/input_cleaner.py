import re


def clean(amr):
    join_model_name(amr)
    # join_al_names(amr)
    # join_possessive_stroke_in_entity(amr)
    split_entity_with_slash(amr)
    split_entity_with_non(amr)
    split_entity_prefix(amr, 'anti')
    split_entity_prefix(amr, 'ex')
    split_entity_prefix(amr, 'cross')
    split_entity_prefix(amr, 'pro')
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


def join_possessive_stroke_in_entity(amr):
    while True:
        span = None
        if len(amr.tokens) < 2:
            break
        for i in range(1, len(amr.tokens)):
            x, y = amr.tokens[i - 1: i + 1]
            if y == "'" and x[0].isupper():
                span = list(range(i - 1, i + 1))
                joined_tokens = ''.join([x, y])
                pos = amr.pos_tags[i]
                ner = amr.ner_tags[i]
                break
        else:
            break
        amr.replace_span(span, [joined_tokens], [pos], [ner])


def split_entity_with_slash(amr):
    # Split named entity word with '/', e.g. 'Romney/McDonnell'.
    while True:
        index = None
        for i, token in enumerate(amr.tokens):
            if token[0].isupper() and '/' in token and token[token.index('/') + 1].isupper():
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        x, y = amr.tokens[index].split('/', 1)
        amr.replace_span([index], [x, '/', y], [pos, 'SYM', pos], [ner, ner, ner])


def split_entity_with_non(amr):
    # Split named entity word with 'non', e.g. 'nonRomney'.
    while True:
        index = None
        for i, token in enumerate(amr.tokens):
            if token.startswith('non') and len(token) > 3 and token[3].isupper():
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        x = amr.tokens[index]
        amr.replace_span([index], ['non', x[3:]], ['JJ', pos], ['O', ner])


def split_entity_prefix(amr, prefix):
    # Split word with 'anti-' prefix.
    while True:
        index = None
        for i, lemma in enumerate(amr.lemmas):
            if lemma.lower().startswith(prefix + '-'):
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        _, lemma = amr.lemmas[index].split('-', 1)
        amr.replace_span([index], [prefix, lemma], ['JJ', pos], [ner, ner])


def replace_NT_dollar_abbr(amr):
    # Replace 'NT' in front of '$' with 'Taiwan'.
    for i, token in enumerate(amr.tokens):
        if token == 'NT' and len(amr.tokens) > i + 1 and amr.tokens[i + 1] in ('$', 'dollars', 'dollar'):
            amr.replace_span([i], ['Taiwan'], ['NNP'], ['COUNTRY'])