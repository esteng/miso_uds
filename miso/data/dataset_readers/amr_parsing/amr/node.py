
class AMRNode:

    attribute_priority = [
        'instance', 'quant', 'mode', 'value', 'name', 'li', 'mod', 'frequency',
        'month', 'day', 'year', 'time', 'unit', 'decade', 'poss'
    ]

    def __init__(self, identifier, attributes=None, copy_of=None):
        self.identifier = identifier
        if attributes is None:
            self.attributes = []
        else:
            self.attributes = attributes
            # self._sort_attributes()
        self._num_copies = 0
        self.copy_of = copy_of

    def _sort_attributes(self):
        def get_attr_priority(attr):
            if attr in self.attribute_priority:
                return self.attribute_priority.index(attr), attr
            if not re.search(r'^(ARG|op|snt)', attr):
                return len(self.attribute_priority), attr
            else:
                return len(self.attribute_priority) + 1, attr
        self.attributes.sort(key=lambda x: get_attr_priority(x[0]))

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if not isinstance(other, AMRNode):
            return False
        return self.identifier == other.identifier

    def __repr__(self):
        ret = str(self.identifier)
        for k, v in self.attributes:
            if k == 'instance':
                ret += ' / ' + v
                break
        return ret

    def __str__(self):
        ret = repr(self)
        for key, value in self.attributes:
            if key == 'instance':
                continue
            ret += '\n\t:{} {}'.format(key, value)
        return ret

    @property
    def instance(self):
        for key, value in self.attributes:
            if key == 'instance':
                return value
        else:
            return None

    @property
    def ops(self):
        ops = []
        for key, value in self.attributes:
            if re.search(r'op\d+', key):
                ops.append((int(key[2:]), value))
        if len(ops):
            ops.sort(key=lambda x: x[0])
        return [v for k, v in ops]

    def copy(self):
        attributes = None
        if self.attributes is not None:
            attributes = self.attributes[:]
        self._num_copies += 1
        copy = AMRNode(self.identifier + '_copy_{}'.format(self._num_copies), attributes, self)
        return copy

    def remove_attribute(self, attr, value):
        self.attributes.remove((attr, value))

    def add_attribute(self, attr, value):
        self.attributes.append((attr, value))

    def replace_attribute(self, attr, old, new):
        index = self.attributes.index((attr, old))
        self.attributes[index] = (attr, new)

    def get_frame_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and re.search(r'-\d\d$', v):
                yield k, v

    def get_senseless_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and not re.search(r'-\d\d$', v):
                yield k, v

