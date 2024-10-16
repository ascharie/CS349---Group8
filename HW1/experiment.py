import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ID3 as decision_tree


data_url = 'HW1/house_votes_84.data'
columns = ['Class'] + ['Vote' + str(i) for i in range(1, 17)]
df = pd.read_csv(data_url, header=None, names=columns)
df = df.replace('?', None)

examples = df.to_dict(orient='records')

def run_experiment(training_sizes, examples, num_trials):
    with_pruning_accuracies = []
    without_pruning_accuracies = []

    for train_size in training_sizes:
        acc_with_pruning = []
        acc_without_pruning = []

        for _ in range(num_trials):

            train_set, test_set = train_test_split(examples, train_size=train_size)

            # without pruning
            tree_without_pruning = decision_tree.ID3(train_set, default=0)
            acc_without_pruning.append(test_accuracy(tree_without_pruning, test_set))

            # with pruning
            tree_with_pruning = decision_tree.ID3(train_set, default=0)

            decision_tree.prune(tree_with_pruning, test_set)

            acc_with_pruning.append(test_accuracy(tree_with_pruning, test_set))

        # Average accuracy for pruning and not
        with_pruning_accuracies.append(sum(acc_with_pruning) / len(acc_with_pruning))
        without_pruning_accuracies.append(sum(acc_without_pruning) / len(acc_without_pruning))

    return with_pruning_accuracies, without_pruning_accuracies

def test_accuracy(tree, test_set):
    '''
    Helper function to calculate accuracy of a tree on the test set.
    '''
    correct_count = 0
    for example in test_set:
        prediction = decision_tree.evaluate(tree, example)
        if prediction == example['Class']:
            correct_count += 1
    return correct_count / len(test_set)

# Training set sizes ranging between 10 and 300 examples
training_sizes = list(range(10, 301, 10))

# Run the experiment with 100 random runs
with_pruning_accuracies, without_pruning_accuracies = run_experiment(training_sizes, examples,100)



# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, with_pruning_accuracies, label='With Pruning', color='blue')
plt.plot(training_sizes, without_pruning_accuracies, label='Without Pruning', color='red')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy on Test Data')
plt.title('Learning Curves: Pruned vs Unpruned Decision Trees')
plt.legend()
plt.grid(True)
plt.show()