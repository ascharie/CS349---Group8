import math
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# returns Euclidean distance between vectors and b
def euclidean(a,b):
    assert(len(a) == len(b))
    dist = math.sqrt(sum((x * x) + (y * y) for x, y in zip(a, b)))
    assert(dist == np.linalg.norm(np.array(a)-np.array(b)))
    return(dist)
        
# returns Cosine Similarity between vectors and b
def cosim(a,b):
    dist = dot(a,b) / (euclidean(a,[0]*len(a)) * euclidean(b,[0]*len(b)))
    return(dist)

# returns dot product of a and b
def dot(a,b):
    assert(len(a) == len(b))
    dot = sum([a[i]*b[i] for i in range(len(a))])
    assert(dot == np.dot(np.array(a), np.array(b)))
    return(dot)

# returns Pearson correlation coefficient between a and b
def pearson(a,b):
    assert(len(a) == len(b))
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum([(a[i] - mean_a) * (b[i] - mean_b) for i in range(len(a))])
    denum = euclidean(a, [mean_a]*len(a)) * euclidean(b, [mean_b]*len(b))
    p_coeff = num / denum
    assert(p_coeff == np.corrcoef(np.array(a), np.array(b))[0][1])
    return(p_coeff)

# returns hamming distance between a and b
# a and b are binary vectors
def hamming(a,b):
    assert(len(a) == len(b))
    dist = [a[i] + b[i] for i in range(len(a))].count(1)
    assert(dist == np.count_nonzero(np.array(a) - np.array(b)))
    return dist

# returns binary reduced data
def binarize(X):
    X_reduced = []
    for i in X:
        X_reduced.append([i[0], [0 if j == '0' else 1 for j in i[1]]])
    return X_reduced

# returns PCA reduced data
def pca(X, query):
    labels_X = [example[0] for example in X]
    X_data = np.array([example[1] for example in X])

    # standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    query_scaled = scaler.transform(query)

    # reduce dimensions
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    X_reduced = [[labels_X[i], list(X_pca[i])] for i in range(len(X_pca))]
    query_reduced = pca.transform(query_scaled)

    return X_reduced, query_reduced

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
    return(labels)

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
    