import math, random
import numpy as np
import random
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# returns Euclidean distance between vectors and b
def euclidean(a,b):
    assert(len(a) == len(b))
    dist = math.sqrt(sum([(x - y)**2 for x, y in zip(a, b)]))
    assert(abs(dist - np.linalg.norm(np.array(a)-np.array(b)))<1e-5)
    
    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):
    dist = dot(a,b) / (euclidean(a,[0]*len(a)) * euclidean(b,[0]*len(b)))
    return(dist)

# returns dot product of a and b
def dot(a,b):
    assert(len(a) == len(b))
    dot = sum([a[i]*b[i] for i in range(len(a))])
    assert(abs(dot - np.dot(np.array(a), np.array(b)))<1e-5)
    return(dot)

# returns Pearson correlation coefficient between a and b
def pearson(a,b):
    assert(len(a) == len(b))
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum([(a[i] - mean_a) * (b[i] - mean_b) for i in range(len(a))])
    denum = euclidean(a, [mean_a]*len(a)) * euclidean(b, [mean_b]*len(b))
    p_coeff = num / denum
    assert(abs(p_coeff - np.corrcoef(np.array(a), np.array(b))[0][1])<1e-5)
    return(p_coeff)

# returns hamming distance between a and b
# a and b are binary vectors
def hamming(a,b):
    assert(len(a) == len(b))
    dist = [a[i] + b[i] for i in range(len(a))].count(1)
    assert(dist == np.count_nonzero(np.array(a) - np.array(b)))
    return dist

# returns binary reduced data
def binarize(X_train, X_valid, X_test):
    X_reduced_vec = []
    for X in [X_train, X_valid, X_test]:
        X_reduced = []
        for i in X:
            X_reduced.append([i[0], [0 if j == '0' else 1 for j in i[1]]])
        X_reduced_vec.append(X_reduced)
    return X_reduced_vec[0], X_reduced_vec[1], X_reduced_vec[2]

# returns data converted to floats
def make_float(X_train, X_valid, X_test):
    X_reduced_vec = []
    for X in [X_train, X_valid, X_test]:
        X_reduced = []
        for i in X:
            X_reduced.append([i[0], [float(j) for j in i[1]]])
        X_reduced_vec.append(X_reduced)
    return X_reduced_vec[0], X_reduced_vec[1], X_reduced_vec[2]

# returns PCA reduced data
def pca(X_train, X_valid, X_test):
    labels_vec = [[example[0] for example in X] for X in [X_train, X_valid, X_test]]
    data_vec = [np.array([example[1] for example in X]) for X in [X_train, X_valid, X_test]]

    # standardize data
    scaler = StandardScaler()
    scaler.fit_transform(data_vec[0]) # fit on train
    data_scaled_vec = [scaler.transform(data) for data in data_vec]

    # reduce dimensions
    pca = PCA(n_components=0.95)
    pca.fit_transform(data_scaled_vec[0]) # fit on train
    data_transformed_vec = [pca.transform(data) for data in data_scaled_vec]

    X_reduced_vec = []
    for i in range(3):
        X_reduced = [[labels_vec[i][j], data_transformed_vec[i][j]] for j in range(len(labels_vec[i]))]
        X_reduced_vec.append(X_reduced)

    return X_reduced_vec[0], X_reduced_vec[1], X_reduced_vec[2]

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    k = 3 
    labels = []

    for q in query:
        distances = []
        for label, attribs in train:
            if metric == "euclidean":
                distance = euclidean(q, attribs)
            elif metric == "cosim":
                distance = cosim(q, attribs)
                
            distances.append((distance, label))
        
        distances.sort(key=lambda x: x[0], reverse=(metric == "cosim"))
        nearest_neighbors = distances[:k]
        nearest_labels = [label for _, label in nearest_neighbors]
        labels.append(Counter(nearest_labels).most_common(1)[0][0])

    return (labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    train_data = [attribs for _, attribs in train] # ignore labels
    k = 10

    # inialize initial cluster means randomly
    cluster_labels = [chr(ord('a') + i) for i in range(k)]
    random_cluster_means = random.sample(train_data, k)
    initial_guess = [[cluster_labels[i], random_cluster_means[i]] for i in range(k)]

    # train kmeans
    trained_means = kmeans_train(train_data, metric, initial_guess)

    # assign labels to query data
    labels = []
    for q in query:
        distances = []
        for label, attribs in trained_means:
            if metric == "euclidean":
                distance = euclidean(q, attribs)
            elif metric == "cosim":
                distance = cosim(q, attribs)
            
            distances.append((distance, label))
        distances.sort(key=lambda x: x[0], reverse=(metric == "cosim"))
        labels.append(distances[0][1])
    return labels

def kmeans_train(train_data, metric, means):
    # assign each data point in train_data to nearest cluster means
    labels = []
    for q in train_data:
        distances = []
        for label, attribs in means:
            if metric == "euclidean":
                distance = euclidean(q, attribs)
            elif metric == "cosim":
                distance = cosim(q, attribs)
            
            distances.append((distance, label))
        distances.sort(key=lambda x: x[0], reverse=(metric == "cosim"))
        labels.append(distances[0][1])
    
    # update cluster means
    new_means = []
    for cluster in range(len(means)):
        label = means[cluster][0]
        cluster_data = [train_data[i] for i in range(len(train_data)) if labels[i] == means[cluster][0]]
        num = []
        # for each element of each entry in cluster_data, sum them up and append to num
        for i in range(len(cluster_data[0])):
            num.append(sum([cluster_data[j][i] for j in range(len(cluster_data))]))
        new_means.append([label, [num[i] / len(cluster_data) for i in range(len(num))]])

    # check for convergence
    if np.linalg.norm(np.array([x[1] for x in means]) - np.array([x[1] for x in new_means])) < 1e-5:
        return new_means
    else:
        return kmeans_train(train_data, metric, new_means)
    

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    