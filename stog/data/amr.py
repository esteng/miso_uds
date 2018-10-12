from collections import OrderedDict
class AMRNode():
    def __init__(self, name, instance):
        self.name = name
        self.instance = instance
        self.children = OrderedDict()
        self.parent = None
        self.relation = None

    def add_children(self,relation, node):
        self.children[relation] = node
        self.children[relation].parent = self

    def set_relation(self, relation):
        self.relation = relation

    def __repr__(self):
        return "(relation : {}, name : {}, instance : {})".format(self.relation, self.name, self.instance)


class AMRTree():
    def __init__(self, string):
        self.root_node = None
        self.node_list = []
        self.id = None
        self._parse_string(string)
        self._cal_corefenrence()

    def _parse_string(self, string):
        stack = []
        current_node = None
        for item in string.split(' '):
            #if item == ":topic":
            #   import pdb;pdb.set_trace()
            if len(stack) > 1 and stack[-1] == "/":
                name = self._strip_token(stack[-2])
                instance = self._strip_token(item)
                new_node = AMRNode(name, instance)
                self._register_node(new_node)

                if self.root_node is None:
                    new_node.set_relation("root")
                    self.root_node = new_node
                    self.root_node.parent = self.root_node
                    current_node = self.root_node

                else:
                    #import pdb; pdb.set_trace()
                    relation = self._strip_token(stack[-3][1:])
                    current_node.add_children(
                        relation,
                        new_node
                    )
                    new_node.set_relation(relation)
                    current_node = self._change_position(item, current_node.children[relation])


                stack = []
                continue


            elif self._is_relation(item) and len(stack) > 0:
                relation = self._strip_token(stack[-2])
                instance = self._strip_token(stack[-1])

                new_node = AMRNode(instance, None)
                new_node.set_relation(relation)

                current_node.add_children(relation, new_node)

                stack = []

                self._register_node(current_node.children[relation])


            elif self._is_end(item) and len(stack) > 0 and self._is_relation(stack[-1]):
                relation = self._strip_token(stack[-1])
                instance = self._strip_token(item)

                new_node = AMRNode(instance, None)
                new_node.set_relation(relation)

                current_node.add_children(relation, new_node)

                stack = []

                self._register_node(current_node.children[relation])

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
            return token[:-(i-1)].replace("\"", "")
        else:
            return token.replace("\"", "")

    def _register_node(self, node):
        node.id = len(self.node_list)
        self.node_list.append(node)

    def _cal_corefenrence(self):
        name_list = [item.name for item in self.node_list]
        self.coref = [-1 for item in self.node_list]
        for idx, node in enumerate(self.node_list):
            if node.name in name_list[:idx]:
                copy_idx = name_list[:idx].index(node.name)
                self.coref[idx] = copy_idx
            if node.instance is None:
                node.instance = node.name if self.coref[idx] == -1 else self.node_list[self.coref[idx]].instance

    def get_names(self):
        return [node.name for node in self.node_list]

    def get_instance(self):
        return [node.instance for node in self.node_list]

    def get_coref(self):
        return self.coref

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
