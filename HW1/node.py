class Node:
    def __init__(self, label = None, attribute = None, is_leaf = False):
        self.label = label
        self.children = {}
        # you may want to add additional fields here...
        self.attribute = attribute #best attribute to split on
        self.is_leaf = is_leaf
        
    def add_child(self, val, child):
        self.children[val] = child