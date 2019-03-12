import json


class AMR:

    def __init__(self,
                 id=None,
                 sentence=None,
                 graph=None,
                 tokens=None,
                 lemmas=None,
                 pos_tags=None,
                 ner_tags=None,
                 abstract_map=None,
                 misc=None):
        self.id = id
        self.sentence = sentence
        self.graph = graph
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.abstract_map = abstract_map
        self.misc = misc

    def is_named_entity(self, index):
        return self.ner_tags[index] not in ('0', 'O')

    def get_named_entity_span(self, index):
        if self.ner_tags is None or not self.is_named_entity(index):
            return []
        span = [index]
        tag = self.ner_tags[index]
        prev = index - 1
        while prev > 0 and self.ner_tags[prev] == tag:
            span.append(prev)
            prev -= 1
        next = index + 1
        while next < len(self.ner_tags) and self.ner_tags[next] == tag:
            span.append(next)
            next += 1
        return span

    def find_span_indexes(self, span):
        for i, token in enumerate(self.tokens):
            if token == span[0]:
                _span = self.tokens[i: i + len(span)]
                if len(_span) == len(span) and all(x == y for x, y in zip(span, _span)):
                    return list(range(i, i + len(span)))
        return None

    def replace_span(self, indexes, new, pos=None, ner=None):
        self.tokens = self.tokens[:indexes[0]] + new + self.tokens[indexes[-1] + 1:]
        self.lemmas = self.lemmas[:indexes[0]] + new + self.lemmas[indexes[-1] + 1:]
        if pos is None:
            pos = [self.pos_tags[indexes[0]]]
        self.pos_tags = self.pos_tags[:indexes[0]] + pos + self.pos_tags[indexes[-1] + 1:]
        if ner is None:
            ner = [self.ner_tags[indexes[0]]]
        self.ner_tags = self.ner_tags[:indexes[0]] + ner + self.ner_tags[indexes[-1] + 1:]

    def remove_span(self, indexes):
        self.replace_span(indexes, [], [], [])

    def __repr__(self):
        fields = []
        for k, v in dict(
            id=self.id,
            snt=self.sentence,
            tokens=self.tokens,
            lemmas=self.lemmas,
            pos_tags=self.pos_tags,
            ner_tags=self.ner_tags,
            abstract_map=self.abstract_map,
            misc=self.misc,
            graph=self.graph
        ).items():
            if v is None:
                continue
            if k == 'misc':
                fields += v
            elif k == 'graph':
                fields.append(str(v))
            else:
                if not isinstance(v, str):
                    v = json.dumps(v)
                fields.append('# ::{} {}'.format(k, v))
        return '\n'.join(fields)

    def get_src_tokens(self):
        return self.lemmas if self.lemmas else self.sentence.split()