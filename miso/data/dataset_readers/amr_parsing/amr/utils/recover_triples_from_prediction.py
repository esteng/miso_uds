import re


def normalize_number(text):
    if re.search(r'^\d+,\d+$', text):
        text = text.replace(',', '')
    return text


def calibrate_multiroots(heads):
    for i in range(1, len(heads)):
        if heads[i] == 0:
            heads[i] = 1
    return heads


def is_attribute_value(value):
    return re.search(r'(^".*"$|^[^a-zA-Z]+$)', value) is not None


def is_attribute_edge(label):
    return label in ('instance', 'mode', 'li', 'value', 'month', 'year', 'day', 'decade', 'ARG6')


def abstract_node(value):
    return re.search(r'^([A-Z]+|DATE_ATTRS|SCORE_ENTITY|ORDINAL_ENTITY)_\d+$', value)


def abstract_attribute(value):
    return re.search(r'^_QUANTITY_\d+$', value)


def recover_triples_from_prediction(prediction):
    nodes = [normalize_number(n) for n in prediction['nodes']]
    heads = calibrate_multiroots(prediction['heads'])
    corefs = [int(x) for x in prediction['corefs']]
    head_labels = prediction['head_labels']

    triples = []
    top = None
    # Build the variable map from variable to instance.
    variable_map = {}
    for coref_index in corefs:
        node = nodes[coref_index - 1]
        head_label = head_labels[coref_index - 1]
        if (re.search(r'[/:\\()]', node) or is_attribute_value(node) or
                is_attribute_edge(head_label) or abstract_attribute(node)):
            continue
        variable_map['vv{}'.format(coref_index)] = node
    for head_index in heads:
        if head_index == 0:
            continue
        node = nodes[head_index - 1]
        coref_index = corefs[head_index - 1]
        variable_map['vv{}'.format(coref_index)] = node

    # Build edge triples and other attribute triples.
    for i, head_index in enumerate(heads):
        if head_index == 0:
            top_variable = 'vv{}'.format(corefs[i])
            if top_variable not in variable_map:
                variable_map[top_variable] = nodes[i]
            top = top_variable
            continue
        head_variable = 'vv{}'.format(corefs[head_index - 1])
        modifier = nodes[i]
        modifier_variable = 'vv{}'.format(corefs[i])
        label = head_labels[i]
        assert head_variable in variable_map
        if modifier_variable in variable_map:
            triples.append((head_variable, label, modifier_variable))
        else:
            # Add quotes if there's a backslash.
            if re.search(r'[/:\\()]', modifier) and not re.search(r'^".*"$', modifier):
                modifier = '"{}"'.format(modifier)
            triples.append((head_variable, label, modifier))

    for var, node in variable_map.items():
        if re.search(r'^".*"$', node):
            node = node[1:-1]
        if re.search(r'[/:\\()]', node):
            parts = re.split(r'[/:\\()]', node)
            for part in parts[::-1]:
                if len(part):
                    node = part
                    break
            else:
                node = re.sub(r'[/:\\()]', '_', node)
        triples.append((var, 'instance', node))

    if len(triples) == 0:
        triples.append(('vv1', 'instance', 'string-entity'))
        top = 'vv1'

    triples.sort(key=lambda x: int(x[0].replace('vv', '')))

    return top, nodes, triples
