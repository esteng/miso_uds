from collections import OrderedDict, defaultdict
import re
class AMRNode():
    def __init__(self, name, instance):
        self.name = name
        self.instance = instance
        self.children = []
        self.parent = None
        self.relation = None
        self.id = None

    def add_children(self,relation, node):
        self.children.append((relation, node))
        self.children[-1][-1].parent = self

    def set_relation(self, relation):
        self.relation = relation

    def __repr__(self):
        return "(relation : {}, name : {}, instance : {})".format(self.relation, self.name, self.instance)


class AMRTree():
    def __init__(self, string="", sent=""):
        self.root_node = None
        self.node_list = []
        self.original_sentence = sent
        if len(string) > 0:
            self._parse_string(string)
            self._cal_corefenrence()

    def _parse_string(self, string):
        stack = []
        current_node = None
        for item in string.split(' '):
            if len(stack) > 1 and stack[-1] == "/":
                name = self._strip_token(stack[-2])
                instance = self._strip_token(item)
                new_node = AMRNode(name, instance)
                self._register_node(new_node)

                if self.root_node is None:
                    new_node.set_relation("root")
                    self.root_node = new_node
                    self.root_node.parent = 0
                    current_node = self.root_node

                else:
                    #import pdb; pdb.set_trace()
                    relation = self._strip_token(stack[-3][1:])
                    current_node.add_children(
                        relation,
                        new_node
                    )
                    new_node.set_relation(relation)
                    current_node = self._change_position(item, current_node.children[-1][-1])


                stack = []
                continue


            elif self._is_relation(item) and len(stack) > 0:
                relation = self._strip_token(stack[-2])
                instance = self._strip_token(stack[-1])

                new_node = AMRNode(instance, None)
                new_node.set_relation(relation)

                current_node.add_children(relation, new_node)

                stack = []

                self._register_node(current_node.children[-1][-1])


            elif self._is_end(item) and len(stack) > 0 and self._is_relation(stack[-1]):
                relation = self._strip_token(stack[-1])
                instance = self._strip_token(item)

                new_node = AMRNode(instance, None)
                new_node.set_relation(relation)

                current_node.add_children(relation, new_node)

                stack = []

                self._register_node(current_node.children[-1][-1])

                current_node = self._change_position(item, current_node)

                continue

            stack.append(item)

    def _change_position(self, token, node):
        if self._is_end(token):
            layer = 1
            while token[-layer] == ")":
                layer += 1

            new_node = node
            for i in range(layer - 1):
                new_node = node.parent
                node = new_node
            return new_node
        else:
            return node

    def _is_relation(self, token):
        return token[0] == ":"

    def _is_end(self, token):
        return token[-1] == ")"

    def _is_begin(self, token):
        return token[0] == "("

    def _strip_token(self, token):
        if self._is_begin(token) or self._is_relation(token):
            return token[1:]
        elif self._is_end(token):
            i = 1
            while token[-i] == ")":
                i = i+1
            return token[:-(i-1)]
        else:
            return token

    def isNodeCoref(self, idx):
        # whether a node is a coreference of another node
        if self.coref[idx] != idx:
            return True
        else:
            return False

    def _register_node(self, node):
        node.id = 1 + len(self.node_list)
        self.node_list.append(node)

    def _get_node_by_idx(self, idx):
        return self.node_list[idx]

    def _cal_corefenrence(self):
        name_list = [item.name for item in self.node_list]
        self.coref = [ii for ii in range(len(self.node_list))]

        for node_idx, node in enumerate(self.node_list):
            if node.name in name_list[:node_idx]:
                copy_node_idx = name_list[:node_idx].index(node.name)
                self.coref[node_idx] = copy_node_idx

            if node.instance is None:
                if self.isNodeCoref(node_idx):
                    node.instance = self._get_node_by_idx(self.coref[node_idx]).instance
                else:
                    node.instance = node.name

    def get_original_sentence(self):
        return self.original_sentence

    def get_names(self):
        return [node.name for node in self.node_list]

    def get_instance(self, bos=None, eos=None):

        instances = [node.instance for node in self.node_list]

        if bos is not None:
            instances = [bos] + instances
        if eos is not None:
            instances = instances + [eos]

        return instances


    def get_coref(self, bos=False, eos=False):
        offset = 0

        coref_indexes = list()

        if bos is True:
            offset = 1
            coref_indexes.append(0)

        coref_indexes += [ x + offset for x in self.coref]

        if eos is True:
            coref_indexes.append(len(coref_indexes))
        
        return coref_indexes


    def get_coref_map(self, bos=False, eos=False):
        # A method to generate data structure to build adjacency field
        coref_indexes = self.get_coref(bos, eos)

        return [
            (token_index, copy_index) for token_index, copy_index in enumerate(coref_indexes)
        ]

            



    def get_relation(self):
        return [node.relation for node in self.node_list]

    def get_parent(self):
        parents = []
        for node in self.node_list:
            if node.parent:
                parents.append(node.parent.id)
            else:
                parents.append(0)
        return parents


    def recover_from_list(self, all_list):
        head_tags = all_list['head_tags']
        head_indices = all_list['head_indices']
        tokens = all_list['tokens']
        self.coref = all_list['coref']

        name_dict = defaultdict(int)

        def get_name_instance(token):
            if token[0] == "\"":
                return token, token

            if re.match('[+-]?([0-9]*[.])?[0-9]+', token):
                return token, None

            letter = token[0] if token[0] != '@' else token[2]

            name_dict[letter] += 1
            if name_dict[letter] > 1:
                return letter + str(name_dict[letter]), token
            else:
                return letter, token


        # add node first, without coref or relations
        for idx, token in enumerate(tokens):
            name, instance = get_name_instance(token)
            self._register_node(AMRNode(name, instance))

        # add coref and relations
        for node_idx, head in enumerate(head_indices):
            current_node = self._get_node_by_idx(node_idx)

            # 1. coref
            if self.isNodeCoref(node_idx):
                current_node.name = self._get_node_by_idx(self.coref[node_idx] - 1).name
                current_node.instance = None

            # 2. relation
            if head_indices[node_idx] == 0:
                self.root_node = current_node
                self.root_node.set_relation('root')

            # find children node
            for child_idx, parent_idx in enumerate(head_indices):
                if parent_idx - 1 == node_idx:
                    child_node = self._get_node_by_idx(child_idx)
                    relation = head_tags[child_idx]
                    child_node.set_relation(relation)
                    current_node.add_children(relation, child_node)

        #print(self.pretty_str())
        #import pdb;pdb.set_trace()

    def pretty_str(self):

        def _print_node(node, level, relation=None):
            if relation == "mode":
                return node.instance

            if len(node.children) == 0:
                if node.instance is None or (node.name == node.instance and node.instance != "i"):
                    return "{}".format(node.name)
                else:
                    return "({} / {})".format(node.name, node.instance)
            else:
                if node.instance:
                    string = "({} / {}".format(node.name, node.instance)
                elif re.match('[+-]?([0-9]*[.])?[0-9]+', node.name):
                    string = "({} / {}".format(node.name, node.name)
                else:
                    string = "{}".format(node.name)

                for relation, child in node.children:
                    string += "\n {} :{} {}".format('\t'*level, relation, _print_node(child, level + 1, relation))

                if node.instance:
                    string += ")"
                return string

        return _print_node(self.root_node, 1)

    def __repr__(self):
        return self.pretty_str()








