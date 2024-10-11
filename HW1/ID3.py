from node import Node
from collections import Counter
import math

def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node) 
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    attributes = list(examples[0].keys())[:]
    attributes.remove('Class')
    
    return id3(examples, attributes, default)
    
def id3(examples, attributes, default):
    if not examples:
        return Node(label=default, is_leaf=True)
    if all(example['Class'] == examples[0]['Class'] for example in examples):
        return Node(label=examples[0]['Class'], is_leaf=True)
    if not attributes:
        most_common_class = Counter(example['Class'] for example in examples if 'Class' in example).most_common(1)[0][0]
        return Node(label=most_common_class, is_leaf=True)
    
    best_attribute = best_split(examples, attributes)
    most_common_class = Counter(example['Class'] for example in examples if 'Class' in example).most_common(1)[0][0]
    t = Node(label=most_common_class, attribute=best_attribute)
    
    splits = split_with(best_attribute, examples)
    unused_attributes = attributes[:]
    unused_attributes.remove(best_attribute)
    
    for value, split in splits.items():
        t.add_child(value, id3(split, unused_attributes, default))
    
    return t        
  
        
def information_gain(examples, attribute):
    original_entropy = entropy(examples)
    
    splits = split_with(attribute, examples)
    new_entropy = sum(entropy(split) * len(split) / len(examples) for split in splits.values())
    
    return original_entropy - new_entropy


def split_with(attribute, examples):
    splits = {}
    
    for example in examples:
        if example[attribute] not in splits:
            splits[example[attribute]] = []
        splits[example[attribute]].append(example)
        
    return splits


def entropy(examples):
    if not examples:
        return 0
    
    class_numbers = Counter(example['Class'] for example in examples)

    return -sum(class_count / len(examples) * math.log2(class_count / len(examples)) for class_count in class_numbers.values())

def best_split(examples, attributes):
    return max(attributes, key = lambda attribute: information_gain(examples, attribute))

def prune(node, examples):
    if node is None or node.label is not None:
        return

    splits = split_with(node.attribute, examples)

    for value, child in node.children.items():
        prune(child, splits[value])
        
    subtree_accuracy = test(node, examples)
    pruned_accuracy = test_pruned(examples)

    if pruned_accuracy >= subtree_accuracy:
        node.children = {}
        node.is_leaf = True

def test_pruned(examples):
    most_common_class = Counter(example['Class'] for example in examples if 'Class' in example).most_common(1)[0][0]
    
    correct_count = 0

    for example in examples:
        if most_common_class == example['Class']:
            correct_count += 1

    return correct_count / len(examples)

def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    correct_count = 0
    for example in examples:
        label = evaluate(node, example)
        if label == example['Class']:
            correct_count += 1
    
    return correct_count / len(examples)


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    if node.is_leaf:
        return node.label
    return evaluate(node.children[example[node.attribute]], example)
