from node import Node
from collections import Counter
import math, random

def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node) 
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    attributes = list(examples[0].keys()) # get the attributes without the class attribute
    attributes.remove('Class')
    
    if not examples: # if there are no examples return a leaf node with the default label
        return Node(label=default, is_leaf=True)
    if len(set([example['Class'] for example in examples])) == 1: # if all examples have the same label return a leaf node with that label 
        return Node(label=examples[0]['Class'], is_leaf=True)
    if not attributes: # if there are no attributes return a leaf node with the most common label
        most_common_class = most_common_value(examples, 'Class')
        return Node(label=most_common_class, is_leaf=True)
    
    best_attribute = best_split(examples, attributes) # get the best attribute to split on
    t = Node(attribute=best_attribute) # create a new node with the instance attribute of best attribute
    splits = split_with(best_attribute, examples) # split the examples on the best attribute
    
    for value, split in splits.items(): # for each attribute value create a child node using ID3()
        sub_split = [{key: value for key, value in example.items() if key != best_attribute} for example in split]
        t.add_child(value, ID3(sub_split, default))
    
    return t
        
def information_gain(examples, attribute):
    '''
    Takes in an array of examples and an attribute, and returns the information gain of the attribute.
    '''
    original_entropy = entropy(examples)
    
    splits = split_with(attribute, examples) # split the examples on the attribute
    conditional_entropy = sum(entropy(split) * len(split) / len(examples) for split in splits.values()) # calculate the conditional entropy of the class label given the attribute value.
    return original_entropy - conditional_entropy # return the information gain


def split_with(attribute, examples):
    '''
    Takes in an attribute and an array of examples, and returns a dictionary containing the split examples.
    '''
    splits = {}
    for example in examples: # split the examples on the attribute and store them in a dictionary
        if example[attribute] not in splits:
            splits[example[attribute]] = []
        splits[example[attribute]].append(example)

    return splits


def entropy(examples):
    '''
    Takes in an array of examples, and returns the entropy of the class label.
    '''
    if not examples:
        return 0
    
    class_counts = Counter(example['Class'] for example in examples) # count of each class label

    return -sum(class_count / len(examples) * math.log2(class_count / len(examples)) for class_count in class_counts.values()) # return entropy

def best_split(examples, attributes):
    '''
    Takes in an array of examples and an array of attributes, and returns the best attribute to split on.
    '''
    return max(attributes, key = lambda attribute: information_gain(examples, attribute))

def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.
    '''
    if node is None or node.is_leaf:
        return
    
    splits = split_with(node.attribute, examples) # split the examples on the attribute of node

    for value, child in node.children.items():  # prune the children of node
        if value in splits:
            prune(child, splits[value])

    subtree_accuracy = test(node, examples) # get the validation set accuracy of the subtree
    pruned_accuracy = test_pruned(examples) # get the validation set accuracy of the most common class label

    if pruned_accuracy >= subtree_accuracy: # prune if the most common class label is more accurate
        node.children = {}
        node.is_leaf = True
        node.label = most_common_value(examples, 'Class')

def test_pruned(examples):
    '''
    Takes in an array of examples, and returns the accuracy of the most common class label.
    '''
    most_common_class = most_common_value(examples, 'Class')

    correct_count = 0

    for example in examples: # count the number of examples with the most common class label.
        if most_common_class == example['Class']:
            correct_count += 1

    return correct_count / len(examples)

def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples. Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    correct_count = 0
    for example in examples:
        try:
           label = evaluate(node, example)
        except:
            node.show()
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
    
    if example[node.attribute] in node.children:
        return evaluate(node.children[example[node.attribute]], example)
    else: # for small datasets not every attribute value is present in data split
       return node.label

def preprocess(examples, mode='remove'):
  '''
  Takes in an array of examples, and returns a preprocessed array of examples. Mode "remove"
  removes examples with missing attribute values. Mode "impute" imputes missing attribute
  values using the most common value for each attribute. Mode "keep" does not modify the examples.
  '''
  proc_examples = examples.copy()

  if mode == "remove": # remove examples with missing attribute values
    for example in proc_examples:
      if '?' in example.values():
        proc_examples.remove(example)
    return proc_examples

  elif mode == "impute": # impute missing attribute values using the most common value for each attribute
    attribute_values = {}
    for key in proc_examples[0].keys():
      attribute_values[key] = most_common_value(proc_examples, key)

    for example in proc_examples: # replace missing values with the most common value
      for key in example.keys():
        if example[key] == '?':
          example[key] = attribute_values[key]
    return proc_examples

  elif mode == "keep": # do not modify the examples
    return proc_examples
  
def most_common_value(examples, target='Class'):
    '''
    Takes in an array of examples, and returns the most common target value.
    '''
    values = [example[target] for example in examples]
    return max(set(values), key=values.count)


#Code for random forest:

def construct_random_forest(examples, default, number=100):
    min_row_extraction = 30
    min_column_extraction = 5
    forest = []
    features = list(examples[0].keys())
    features.remove('Class')
    
    for i in range(number):
        num_of_rows = random.randint(min_row_extraction, len(examples))
        num_of_columns = random.randint(min_column_extraction, len(features))
        training_data = random.sample(examples, num_of_rows)
        training_data = extract_columns(training_data, num_of_columns, features)
    
        tree = ID3(training_data, default)
        forest.append(tree)
        
    return forest

def extract_columns(data, num_of_columns, features):
    sampled_attributes = random.sample(features, num_of_columns)
    sampled_attributes.append('Class')
    
    return [{key: value for key, value in example.items() if key in sampled_attributes} for example in data]

def evaluate_random_forest(forest, example):
    predictions = []
    for tree in forest:
        predictions.append(evaluate(tree, example))
        
    return Counter(predictions).most_common(1)[0][0]

def test_random_forest(forest, examples):
    correct_count = 0
    for example in examples:
        label = evaluate_random_forest(forest, example)
    
        if label == example['Class']:
            correct_count += 1
    
    return correct_count / len(examples)