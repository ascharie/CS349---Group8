from classes import Node
from collections import Counter
import math, random

def ID3(X, y, default):
    attributes = list(X.columns)

    if X.empty: # if there are no more examples return a leaf node with the default label
        return Node(label=default, is_leaf=True)
    if y.nunique() == 1: # if all examples have the same label return a leaf node with that label
        return Node(label=y.iloc[0], is_leaf=True)
    if not attributes: # if there are no attributes return a leaf node with the most common label
        most_common_class = y.value_counts().idxmax()
        return Node(label=most_common_class, is_leaf=True)
    
    best_attribute = best_split(X, y, attributes) # get the best attribute to split on
    t = Node(label=default, attribute=best_attribute) # create a new node with the instance attribute of best attribute
    splits = split_with(best_attribute, X, y) # split the examples on the best attribute
    
    for value, X_split, y_split in splits: # for each attribute value create a child node using ID3()
        most_common_class = y_split.value_counts().idxmax()
        t.add_child(value, ID3(X_split.drop(columns=[best_attribute]), y_split, most_common_class))
    return t

        
def information_gain(X, y, attribute):
    '''
    Takes in X, y, and an attribute, and returns the information gain of the attribute.
    '''
    original_entropy = entropy(y)
    
    splits = split_with(attribute, X, y) # split the data on the attribute

    conditional_entropy = 0
    for _, _, y_split in splits:
        conditional_entropy += len(y_split) / len(y) * entropy(y_split)

    return original_entropy - conditional_entropy # return the information gain


def split_with(attribute, X, y):
    splits = []
    for value in X[attribute].unique():
        X_split = X[X[attribute] == value]
        y_split = y.loc[X_split.index]
        splits.append((value, X_split, y_split))

    return splits


def entropy(y):
    '''
    Takes in an array of labels, and returns the entropy.
    '''
    if len(y) == 0:
        return 0
    
    class_counts = y.value_counts() # count of each class label


    sum = 0
    for class_count in class_counts:
        sum += class_count / len(y) * math.log2(class_count / len(y))

    return -sum # return entropy

def best_split(X, y, attributes):
    '''
    Takes in X, y, and an array of attributes, and returns the best attribute to split on.
    '''
    return max(attributes, key = lambda attribute: information_gain(X, y, attribute))

def prune(node, X, y):
    '''
    Takes in a trained tree and a validation set of examples.
    '''
    if node is None or node.is_leaf:
        return
    
    if node.attribute not in X.columns:
        return
    
    splits = split_with(node.attribute, X, y) # split the examples on the attribute of node 

    for value, child in node.children.items():  # prune the children of node
        split = None
        for v, X_split, y_split in splits:
            if v == value:
                split = (X_split, y_split)
                break
        if split is not None:
            X_split, y_split = split
            prune(child, X_split, y_split)

    _, subtree_accuracy = test(node, X, y) # get the validation set accuracy of the subtree
    pruned_accuracy = test_pruned(y) # get the validation set accuracy of the most common class label

    if pruned_accuracy >= subtree_accuracy: # prune if the most common class label is more accurate
        node.children = {}
        node.is_leaf = True
        node.label = most_common_value(y)

def test_pruned(y):
    '''
    Takes in an array of examples, and returns the accuracy of the most common class label.
    '''
    most_common_class = most_common_value(y)

    correct_count = 0

    correct_count = (y == most_common_class).sum()
    return correct_count / len(y)

def test(node, X, y):
    '''
    Takes in a trained tree and a test set of examples. Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    correct_count = 0
    labels = [None] * len(y)
    counter = 0
    for index, example in X.iterrows():
        label = evaluate(node, example)
        if label == None:
            node.show()
            raise ValueError("Label is None")
        if label == y[index]:
            correct_count += 1
        labels[counter] = label
        counter += 1
    
    return labels, correct_count / len(y)


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
  
def most_common_value(series):
    if series.empty:
        raise ValueError("Series is empty")
    return series.value_counts().idxmax()


# Code for random forest:

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