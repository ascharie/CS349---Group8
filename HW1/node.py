class Node:
    def __init__(self, label = None, attribute = None, is_leaf = False):
        self.label = label
        self.children = {}
        self.attribute = attribute # best attribute to split on
        self.is_leaf = is_leaf
        
    def add_child(self, val, child):
        self.children[val] = child

    # def show(self, level=0):
    #     indent = '     ' * level

    #     if not self.is_leaf:
    #         print(indent, 'attribute:', self.attribute)
    #         for value, child in self.children.items():
    #             print(indent, '--', value)
    #             child.show(level + 1)
    #     else:
    #         print(indent, 'label:', self.label)
