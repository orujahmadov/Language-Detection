# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:58:40 2017

@author: Ryan @ ORUJ
"""
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import sys
from ast import literal_eval
import pickle

all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      
FILENAME = 'cleaned_data_final.csv'

#%%
###################### Define USER Functions
        
 # Merge Function
        # data procssing step
        # used to merge features and class of train and test data       
def merge_data(X_train, y_train, X_test, y_test):
    
    Y_train = pd.Series(y_train, name='Y') 
    Y_test = pd.Series(y_test, name='Y')
    TRAIN = pd.concat([pd.DataFrame(X_train),pd.DataFrame(Y_train)],axis=1)
    TEST = pd.concat([pd.DataFrame(X_test),pd.DataFrame(Y_test)],axis=1)
    return TRAIN, TEST

#%%
# Count classses function
    # returns dictionary with all the types of classes and their respective counts
def count_classes(data):
    
    class_counts = dict()                   # create empty dictionary
    data = pd.DataFrame(data)               # maintain data as DataFrame
    for index in data.index:                # cycle through class data
        class_type = data.loc[index]['Y']     # extraxt class
        if class_type not in class_counts:    # if not already added to dictionary
            class_counts[class_type] = 1      # add with count of 1
        else:                               # if already added to dictionary 
            class_counts[class_type] += 1     # increment count
    return class_counts

#%%
# Split function
    # returns a list that returns True when subjected to the Separator 
        #and another that return False when subjected to teh Separator
def split(data, separator):
    
    true_split = false_split = pd.DataFrame()   # create 2 empty dataframes
     pd.DataFrame()           
    for index in data.index:                    # cycle through each row of data
        if separator.check(data.loc[index]):      # check if True when subjected to the Separator 
            true_split = true_split.append(data.loc[index])    # append to True list
            # check if false when subjected to the Separator 
        else:
            false_split = false_split.append(data.loc[index])   # append to False list
    return true_split, false_split

#%%
# Scoring Function
        # Returns gini coefficient where 0 is purely one class in split
                            # where 0.5 is a 50/50 split
                            # the lower the better
def scoring(data):

    class_counts = count_classes(data)          # find all unique classes and there counts
    score = 1                                   # Start with worse score for initialization
    for class_type in class_counts:             # Cycle through unique classes
        class_prob = class_counts[class_type] / float(len(data))    
        score -= class_prob**2
    return score

#%%
# Update Score Function
    # returns updated score with previous scores and split in consideration
    # the higher the better (information gain)
def update_scoring(true_split, false_split, current_score):
    # proportion of data in true split vs. all data(true and false splits)
    proportion = float(len(true_split)) / (len(true_split) + len(false_split))
    # recalculated score based on previous score and split proportions
    updated_score = current_score - proportion * scoring(true_split) - (1 - proportion) * scoring(false_split)
    return updated_score

#%%
# Best Node Split function
    # Evaluates best split and returns best score and best separator feature and value 
def best_node_split(data):
    
    best_score = None                                       # init empty best score
    best_separator = None                         # init empty best seperator
    current_score = scoring(data)                             # calculate first score
    feature_quantity = len(data.loc[data.index[0]]) - 1     # counts number of features

    for feature in range(feature_quantity):                 # cycle through features

        values = set(data[feature])                     # extract unique values of that feature

        for value in values:                                # cycle through values of features
            separator = Separator(feature, value)           # find separator
            true_split, false_split = split(data, separator) # split with separator

            if len(true_split) == 0 or len(false_split) == 0: # break out if either split is empty (no information gain)
                continue

            updated_score = update_scoring(true_split, false_split, current_score) # update score with update scoring function

            if updated_score >= best_score:         # check if updated score exceeds best score
                best_score = updated_score          # if so update best score 
                best_separator = separator          # and best separator
                
    return best_score, best_separator
#%%
# grow tree function
    # function to grow tree out from split to decision node
def grow_tree(data):
    
    score, separator = best_node_split(data)        # find
    
    if score == None:                                  # check if split data is pure (only one class present)
        return Leaf_Node(data)                      # if so make into a leaf node
    else:                                           # otherwise
        true_split, false_split = split(data, separator)    # split data 
        true_branch = grow_tree(true_split)   # recursively grow true branch 
        false_branch = grow_tree(false_split) # recursively grow false branch

    return Decision_Node(separator, true_branch, false_branch)
#%%
# Classification function
        # returns prediction of classification in % likelyhood of being in each class
def classification(data_line, node):

    if isinstance(node, Leaf_Node):     # check to see if node is terminal
        return node.class_prediction    # if so return predictions
    else:                               # otherwise
        if node.separator.check(data_line):  # check seperator 
            return classification(data_line, node.true_branch)  # classify true branch 
        else:
            return classification(data_line, node.false_branch) #  classify false branch
#%%
########################## Define Classes

# Separator class
    # used to evaluate features and values for spliting tree
class Separator:
    
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def check(self, test):# return true or false when evaluated by separator
        response = (test[self.feature] >= self.value)   
        return response
#%%
# Terminal Leaf Node Class
        # returns the prediction which is the probabilities of the data belonging to each class
class Leaf_Node:
    
    def __init__(self, data):
        
        self.class_prediction = count_classes(data)
        
#%%
#   Split Decision Node Class
        # for use in recursion

class Decision_Node:
    
    def __init__(self, separator, true_branch, false_branch):
        self.separator = separator
        self.true_branch = true_branch
        self.false_branch = false_branch

#%%
################### ORUJ'S CODE for data Preprocessing

def get_letters_count(string):
    counter = 0
    for character in all_characters:
        counter+=string.lower().count(character)
        
    return counter
        
# CLEANING
def clean_data(input_x, y):
    cleaned_inputs = []
    cleaned_labels = []
    cleaned_data = []
    for input_feature, label in zip(input_x, y):
        if isinstance(input_feature, str):
            if (get_letters_count(input_feature) > 40 and get_letters_count(input_feature) < 100):
                cleaned_inputs.append(input_feature) 
                cleaned_labels.append(label)
    
    cleaned_data.append(cleaned_inputs)
    cleaned_data.append(cleaned_labels)
    
    return cleaned_data
    
# Feature Extraction
def get_letters_frequency(input_string):
    letters_frequency = []
    for character in all_characters:
        letters_frequency.append(input_string.lower().count(character))
    return letters_frequency

def extract_features(input_data):
    features_array = []
    for string in input_data:
        if isinstance(string, str):
            features_array.append(get_letters_frequency(string))
        else:
            features_array.append(get_letters_frequency(" "))
    return features_array

def preprocess_data(file_train_x, file_train_y):
    # Importing the dataset
    dataset_x = pd.read_csv("data/train_set_x.csv")
    X = dataset_x.iloc[:,1]
    dataset_y = pd.read_csv('data/train_set_y.csv')
    Y = dataset_y.iloc[:,1]
    
    cleaned_data = clean_data(X,Y)
    
    cleaned_X = cleaned_data[0]
    cleaned_Y = cleaned_data[1]
    
    extracted_features = extract_features(cleaned_X)
    
    preprocessed_data = {}
    preprocessed_data['X'] = extracted_features
    preprocessed_data['Y'] = cleaned_Y
    
    return preprocessed_data

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

#%%
if __name__ == '__main__':

    if (len(sys.argv) <3):
        print("Arguments missing")
    else:
        processed_data = sys.argv[1]
        output_model = sys.argv[2]
    
        # PART 1 -> DATA PREPROCESSING
        data = import_preprocessed_data(processed_data)
        X = data['X']
        Y = data['Y']
        # Splitting data to train set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)   
        
        # Merge Features and Class into one DataFrame for both Train and Test data for use in Decision Tree
        TRAIN, TEST = merge_data(X_train, y_train, X_test, y_test)
        
    #%% Grow Tree 
    
        TD = TRAIN 
        MT = grow_tree(TD)
    
    #%% Get Prediction
        
        TTD = TEST[0:500]
        predictions = list()
        for index in TTD.index:
            prediction = classification(TTD.loc[index], MT).keys()[0]
            predictions.append(prediction)
            
    #%% Accuracy Score
            
        from sklearn.metrics import accuracy_score    
        ACC = accuracy_score(list(TTD['Y'][0:500]),predictions)
        print("Decision Tree Clasifier Accuracy Score is " + str(ACC))
        
        pickle.dump(MT, open(output_model, 'wb'))
