# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:17:57 2017

"""

# -*- coding: utf-8 -*-
# Classification template

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from ast import literal_eval
import numpy as np
from sklearn import metrics
import pickle
    
all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      

def replace_elements(list_a, target):
    return [1 if x==target else 0 for x in list_a]

def import_preprocessed_data(file_preprocessed_data):
    preprocessed_data = pd.read_csv(file_preprocessed_data)
    preprocessed_data.X = preprocessed_data.X.apply(literal_eval)
    X = preprocessed_data.iloc[:,0]
    Y = preprocessed_data.iloc[:,1]
    X = X.tolist()
    Y= Y.tolist()
    
    data = {}
    data['X'] = X
    data['Y'] = Y
    
    return data

class LogisticRegression(object):
    
    def __init__(self, X = None, Y = None, learning_rate = 0.005):
        self.X = X
        self.Y = Y
        self.W = Y.zeros((n_in, n_out))         # initialize W 0
        self.b = np.zeros(Y.shape())          # initialize bias 0

        # self.params = [self.W, self.b]
        
    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))

    def train(self, features, target, num_steps, learning_rate, add_intercept = False):
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
    
        weights = np.zeros(features.shape[1])
    
        for step in xrange(num_steps):
            scores = np.dot(features, weights)
            predictions = self.sigmoid(scores)
    
            # Update weights with gradient
            output_error_signal = target - predictions
            gradient = np.dot(features.T, output_error_signal)
            weights += learning_rate * gradient
    
        return weights
    
    def predict(self, X):
        w = self.W
        
        sigmoid_score = []
        predictions = []
        
        for x_input in X:
    
            l1 = w[0][0] + w[0][1:] * x_input
            sigmoid_score[0] = 1 / float(1 + np.exp(-l1))
        
            l2 = w[1][0] + w[1][1:] * x_input
            sigmoid_score[1] = 1 / float(1 + np.exp(-l2))
        
            l3 = w[2][0] + w[2][1:] * x_input
            sigmoid_score[2] = 1 / float(1 + np.exp(-l3))
        
            l4 = w[3][0] + w[3][1:] * x_input
            sigmoid_score[3] = 1 / float(1 + np.exp(-l4))
        
            l5 = w[4][0] + w[4][1:] * x_input
            sigmoid_score[4] = 1 / float(1 + np.exp(-l5))
        
            predictions.append(np.argmax(sigmoid_score))
    
        return predictions
    
if __name__ == "__main__":

    if (len(sys.argv) <3):
        print("Arguments missing")
    else:
        processed_data = sys.argv[1]
        output_model = sys.argv[2]
        
        # PART 1 -> Importing Preprocessed data
        data = import_preprocessed_data(processed_data)
        X = data['X']
        Y = data['Y']
        
        # Splitting data to train set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
        
        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        
        classifier = LogisticRegression()
        classifier.train(X_train, y_train, 30000, 0.0005)
        accuracy = metrics.accuracy_score(y_test, classifier.predict(X_test))
        print("Accuracy score is " + str(accuracy))
        # Save model to specified file
        pickle.dump(classifier, open(output_model, 'wb'))
        
        

    
    
    
