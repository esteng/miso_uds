import re


def clean(amr):
    correct_errors(amr)
    # Named entity
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
    # Date
    split_date_duration(amr)
    split_numerical_date(amr)
    split_year_month(amr)
    split_era(amr)
    split_911(amr)


def correct_errors(amr):
    while True:
        index = None
        for i, token in enumerate(amr.tokens):
            if token == '570000':
                index = i
                tokens = ['2005', '07']
                pos = ['CD', 'CD']
                ner = ['DATE', 'DATE']
                break
            if token == '990000':
                index = i
                tokens = ['1999'] if amr.id.startswith('PROXY_AFP_ENG') else ['1990']
                pos = ['CD']
                ner = ['DATE']
                break
            if token == '860000':
                index = i
                tokens = ['1986']
                pos = ['CD']
                ner = ['DATE']
                break
            if token == '-20040824':
                index = i
                tokens = ['2004', '07', '24']
                pos = ['CD', 'CD', 'CD']
                ner = ['DATE', 'DATE', 'DATE']
                break
            if amr.id.startswith('PROXY_XIN_ENG_20030709_0070.6') and token == 'July':
                index = i
                tokens = ['June']
                pos = ['NNP']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_APW_ENG_20080826_0891.5') and token == 'August':
                index = i
                tokens = ['July']
                pos = ['NNP']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_LTW_ENG_20070514_0055.19') and token == 'May':
                index = i
                tokens = ['March']
                pos = ['NNP']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070430_0038.8') and token == 'February':
                index = i
                tokens = ['April']
                pos = ['NNP']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070504_0296.10') and token == '070513':
                index = i
                tokens = ['20130513']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070504_0296.10') and token == '070514':
                index = i
                tokens = ['20130514']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070607_0366.8') and token == 'April':
                index = i
                tokens = ['June']
                pos = ['NNP']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070612_0538.6') and token == 'June':
                index = i
                tokens = ['December']
                pos = ['NNP']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070612_0538.6') and token == '12':
                index = i
                tokens = ['6']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070620_0032.14') and token == 'June':
                index = i
                tokens = ['6']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070906_0523') and token == 'September':
                index = i
                tokens = ['9']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20070910_0544') and token == 'September':
                index = i
                tokens = ['9']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20071204_0145.25') and token == '200':
                index = i
                tokens = ['2000']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_AFP_ENG_20071206_0630.5') and token == 'November':
                index = i
                tokens = ['10']
                pos = ['CD']
                ner = ['DATE']
                break
            if amr.id.startswith('PROXY_APW_ENG_20080112_0264.5') and token == '080112':
                index = i
                tokens = ['20081112']
                pos = ['CD']
                ner = ['DATE']
                break
        else:
            break
        amr.replace_span([index], tokens, pos, ner)


def join_model_name(amr):
    # Joint the words starting with a cap letter which is followed by '^-\d+$'
    while True:
        span = None
        if len(amr.tokens) < 2:
            break
        for i in range(len(amr.tokens) - 1):
            x, y = amr.tokens[i: i + 2]
            if x.isalpha() and x.isupper() and re.search(r'^-\d+$', y):
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
            if (len(token) and token[0].isupper() and '/' in token and
                token.index('/') + 1 < len(token) and
                token[token.index('/') + 1].isupper()
            ):
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
        if lemma == '':
            amr.replace_span([index], [prefix], ['JJ'], ['O'])
        else:
            amr.replace_span([index], [prefix, lemma], ['JJ', pos], [ner, ner])


def split_date_duration(amr):
    # 201005-201006
    while True:
        index = None
        x = None
        for i, lemma in enumerate(amr.lemmas):
            if re.search(r'^-\d{8}$', lemma) or re.search(r'^-\d{6}$', lemma):
                index = i
                _, x = lemma.split('-')
                break
        else:
            break
        amr.replace_span([index], [x], ['CD'], ['DATE'])



def split_numerical_date(amr):
    # Split the numerical date, e.g. 20080710.
    while True:
        index = None
        year, month, day = None, None, None
        for i, lemma in enumerate(amr.lemmas):
            if (re.search(r'^\d{8}$', lemma) and
                1000 < int(lemma[:4]) < 2100 and  # year
                0 < int(lemma[4:6]) < 13 and  # month
                0 < int(lemma[6:]) < 32  # day
            ):
                index = i
                year, month, day = int(lemma[:4]), int(lemma[4:6]), int(lemma[6:])
                month = '{:02d}'.format(month)
                day = '{:02d}'.format(day)
                break
            elif (re.search(r'^\d{5}$', lemma) and
                    0 < int(lemma[1:3]) < 13 and  # month
                    0 < int(lemma[3:]) < 32  # day
            ):
                index = i
                year, month, day = '0' + lemma[0], int(lemma[1:3]), int(lemma[3:])
                month = '{:02d}'.format(month)
                day = '{:02d}'.format(day)
                break
            elif (re.search(r'^\d{6}$', lemma) and
                    0 < int(lemma[2:4]) < 13 and  # month
                    0 <= int(lemma[4:]) < 32  # day
            ):
                index = i
                year = int(lemma[:2])
                month, day = int(lemma[2:4]), int(lemma[4:])
                year = '{:02d}'.format(year)
                month = '{:02d}'.format(month)
                day = '{:02d}'.format(day)
                break
            elif re.search(r'^\d+/\d+/\d+$', lemma):
                index = i
                year, month, day = lemma.split('/')
                break
            elif re.search(r'^\d+-/\d+-/\d+$', lemma):
                index = i
                year, month, day = lemma.split('-')
                break
        else:
            break
        pos = 'CD'
        ner = 'DATE'
        amr.replace_span([index], [str(year), str(month), str(day)], [pos] * 3, [ner] * 3)


def split_year_month(amr):
    while True:
        index = None
        year, month = None, None
        for i, token in enumerate(amr.tokens):
            m = re.search(r'^(\d+)/(\d+)-*$', token)
            if m:
                index = i
                year, month = m.group(1), m.group(2)
                break
            m = re.search(r'^(\d{4})(\d{2})00$', token)
            if m:
                index = i
                year, month = m.group(1), m.group(2)
                break
        else:
            break
        amr.replace_span([index], [year, month], ['CD', 'CD'], ['DATE', 'DATE'])


def split_era(amr):
    while True:
        index = None
        year, era = None, None
        for i, token in enumerate(amr.tokens):
            if re.search(r'^\d{4}BC$', token):
                index = i
                year, era = token[:4], token[4:]
                break
        else:
            break
        amr.replace_span([index], [year, era], ['CD', 'NN'], ['DATE', 'DATE'])


def split_911(amr):
    while True:
        index = None
        for i, token in enumerate(amr.tokens):
            if token == '911':
                index = i
                break
        else:
            break
        amr.replace_span([index], ['09', '11'], ['CD', 'CD'], ['DATE', 'DATE'])


def replace_NT_dollar_abbr(amr):
    # Replace 'NT' in front of '$' with 'Taiwan'.
    for i, token in enumerate(amr.tokens):
        if token == 'NT' and len(amr.tokens) > i + 1 and amr.tokens[i + 1] in ('$', 'dollars', 'dollar'):
            amr.replace_span([i], ['Taiwan'], ['NNP'], ['COUNTRY'])


