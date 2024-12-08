import math, random
import numpy as np
import random
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
def pca(X_train, X_valid, X_test, var):
    labels_vec = [[example[0] for example in X] for X in [X_train, X_valid, X_test]]
    data_vec = [np.array([example[1] for example in X]) for X in [X_train, X_valid, X_test]]

    # standardize data
    normalizer = StandardScaler()
    normalizer.fit_transform(data_vec[0]) # fit on train
    data_scaled_vec = [normalizer.transform(data) for data in data_vec]

    # reduce dimensions
    pca = PCA(n_components=var) # % variance retained
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

# returns labels of query data set based on closest observation in target data set
def nearest_neighbor(target, query, metric):
    labels = []
    for q in query:
        distances = []
        for label, attribs in target:
            if metric == "euclidean":
                distance = euclidean(q, attribs)
            elif metric == "cosim":
                distance = cosim(q, attribs)
            
            distances.append((distance, label))
        distances.sort(key=lambda x: x[0], reverse=(metric == "cosim"))
        labels.append(distances[0][1])
    return labels

# returns the means trained on the train data set
def kmeans_train(train_data, metric, means):
    # assign each data point in train_data to nearest cluster means
    labels = nearest_neighbor(means, train_data, metric)
 
    # update cluster means
    new_means = []
    for cluster in range(len(means)):
        label = means[cluster][0]
        cluster_data = [train_data[i] for i in range(len(train_data)) if labels[i] == label]
        num = []
        # for each element of each entry in cluster_data, sum them up and append to num
        for i in range(len(cluster_data[0])):
            num.append(sum([cluster_data[j][i] for j in range(len(cluster_data))]))
        new_means.append([label, [num[i] / len(cluster_data) for i in range(len(num))]])

    # check for convergence
    if math.sqrt(sum([x**2 for x in [euclidean(x[1], y[1]) for x,y in zip(means, new_means)]])) < 1e-5:
        return new_means
    else:
        return kmeans_train(train_data, metric, new_means)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# metric is a string specifying either "euclidean" or "cosim".  
def kmeans_evaluate_collaborative_filter(query, means, metric):
    # assign labels to query data
    labels = nearest_neighbor(means, query, metric)
    return labels


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

#collaborative filtering
    
def get_similar_users(train_ratings, movielens_ratings, target_user_id, metric, K):
    distances = defaultdict(float)
    for user, user_ratings in movielens_ratings.items():
        if user != target_user_id:
            #find common movies for target_user_id and user in movielens_ratings
            common_movies = set(train_ratings[target_user_id].keys()).intersection(set(user_ratings.keys()))
            if common_movies:
                a = [train_ratings[target_user_id][movie] for movie in common_movies]
                b = [user_ratings[movie] for movie in common_movies]
                if metric == "euclidean":
                    distance = euclidean(a, b)
                elif metric == "cosim":
                    distance = cosim(a, b)
                # append distance and similar user to distances
                distances[user] += distance
    #find the K most similar users
    sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=(metric == "cosim"))
    return [user for user, _ in sorted_distances[:K]]

def recommend_movies(movielens_ratings, target_user_id, similar_users, M):
    movie_scores = defaultdict(float)
    for user in similar_users:
        for movie, rating in movielens_ratings[user].items():
            #get the movie that target_user_id has not watched
            if movie not in movielens_ratings[target_user_id]:
                #add up the ratings of the movie from similar users
                movie_scores[movie] += rating
    #sort the movies in descending order and recommend M movies with the highest scores
    recommended_movies = sorted(movie_scores, key=movie_scores.get, reverse=True)[:M]
    return recommended_movies

def evaluate_collaborative_filter(recommendations, user_preference):
    #y_true is 1 if the user rated the movie higher than 3, 0 otherwise
    y_true = [1 if rating > 3 else 0 for movie, rating in user_preference.items()]
    #y_pred is 1 if the movie is in recommendations, 0 otherwise
    y_pred = [1 if movie in recommendations else 0 for movie in user_preference]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def read_ratings(file_name):
    ratings = defaultdict(dict)
    with open(file_name, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            user, movie, rating, title, genre, age, gender, occupation = line.strip().split('\t')
            ratings[user][movie] = float(rating)
    return ratings

# part2 question 2
def read_more_data(file_name):
    ratings, user_data, original_ratings = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    movie_genres = defaultdict(dict)
    ages = []

    with open(file_name, 'r') as f:
        next(f)
        for line in f:
            user, movie, rating, _ , genre, age, gender, _ = line.strip().split('\t')
            ratings[user][movie] = original_ratings[user][movie] = float(rating)
            user_data[user] = {'gender': gender, 'age': int(age)}
            ages.append(int(age))
            movie_genres[movie] = set(genre.split('|'))

    # Normalize ages and ratings using Min-Max
    normalizer = MinMaxScaler()
    normalized_ages = normalizer.fit_transform(np.array(ages).reshape(-1, 1)).flatten()
    for i, user in enumerate(user_data):
        user_data[user]['age'] = normalized_ages[i]
        user_ratings = list(ratings[user].values())
        normalized_ratings = normalizer.fit_transform(np.array(user_ratings).reshape(-1, 1)).flatten()
        ratings[user] = dict(zip(ratings[user], normalized_ratings))

    return ratings, user_data, original_ratings, movie_genres

def get_similar_users_improved(train_ratings, movielens_ratings, train_userdata, movielens_userdata, train_genres, movielens_genres, target_user_id, metric, K):
    distance_sums = defaultdict(float)

    target_user_movies = train_ratings[target_user_id].keys()
    target_user_genres = {genre for movie in target_user_movies for genre in train_genres.get(movie,set())}

    for user, user_ratings in movielens_ratings.items():
        if user != target_user_id:
            a_age = train_userdata[target_user_id]['age']
            b_age = movielens_userdata[user]['age']
            a_gender = train_userdata[target_user_id]['gender']
            b_gender = movielens_userdata[user]['gender']

            # add weight (tune based on performance)
            age_weight = 1
            gender_weight = 1
            rating_weight = 1
            genre_weight = 1

            distance_sums[user] += age_weight * euclidean([a_age], [b_age]) + gender_weight * (a_gender != b_gender)
            
            # Find common movies for target_user_id and user in movielens_ratings
            common_movies = set(train_ratings[target_user_id].keys()).intersection(set(user_ratings.keys()))
            if common_movies:
                a_rating = [train_ratings[target_user_id][movie] for movie in common_movies]
                b_rating = [user_ratings[movie] for movie in common_movies]
                
                if metric == "euclidean":
                    distance = euclidean(a_rating, b_rating)
                elif metric == "cosim":
                    distance = cosim(a_rating, b_rating)
                # Accumulate distance for the user
                distance_sums[user] += rating_weight * distance

            # calculate genre similarity between target user and current user
            current_user_genres = {genre for movie in user_ratings.keys() for genre in movielens_genres.get(movie,set())}
            genre_similarity_score = len(target_user_genres.intersection(current_user_genres)) / len(target_user_genres) if target_user_genres else 0

            distance_sums[user] += genre_similarity_score * genre_weight

    # Convert the dictionary to a list of tuples and sort it
    distances = sorted(distance_sums.items(), key=lambda x: x[1], reverse=(metric == "cosim"))
    # Find the K most similar users
    similar_users = [user for user, _ in distances[:K]]
    return similar_users

# end of collaborative filtering
        
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
    