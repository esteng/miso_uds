import json

import penman


class AMRAnnotation:

    def __init__(self,
                 id=None,
                 sentence=None,
                 graph=None,
                 tokens=None,
                 lemmas=None,
                 pos_tags=None,
                 ner_tags=None,
                 misc=None):
        self.id = id
        self.sentence = sentence
        self.graph = graph
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.misc = misc

    def __repr__(self):
        fields = []
        for k, v in dict(
            id=self.id,
            snt=self.sentence,
            tokens=self.tokens,
            lemmas=self.lemmas,
            pos_tags=self.pos_tags,
            ner_tags=self.ner_tags,
            misc=self.misc,
            graph=self.graph
        ).items():
            if v is None:
                continue
            if k == 'misc':
                fields += v
            elif k == 'graph':
                fields.append(penman.encode(v, indent=6))
            else:
                if not isinstance(v, str):
                    v = json.dumps(v)
                fields.append('# ::{} {}'.format(k, v))
        return '\n'.join(fields)


class AMRProcessor:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            amr = AMRAnnotation()
            graph_lines = []
            misc_lines = []
            for line in f:
                line = line.rstrip()
                if line == '':
                    if len(graph_lines) != 0:
                        amr.graph = penman.decode(' '.join(graph_lines))
                        amr.misc = misc_lines
                        yield amr
                        amr = AMRAnnotation()
                    graph_lines = []
                    misc_lines = []
                elif line.startswith('# ::'):
                    if line.startswith('# ::id '):
                        amr.id = line
                    elif line.startswith('# ::snt '):
                        amr.sentence = line[len('# ::snt '):]
                    else:
                        misc_lines.append(line)
                else:
                    graph_lines.append(line)

            if not amr.is_empty():
                amr.graph = penman.decode(' '.join(graph_lines))
                amr.misc = misc_lines
                yield amr

    @staticmethod
    def write(amr_instances, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for amr in amr_instances:
                f.write(str(amr) + '\n\n')
