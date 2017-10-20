#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:43:57 2017

@author: orujahmadov
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import csv
import sys
import pickle


all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      


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

def export_kaggle_results(file_name, header1_name, header2_name, results):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([header1_name, header2_name])
        index = 0
        for result in results:
            filewriter.writerow([index,result])
            index+=1

if __name__ == "__main__":
    
    model_name = ""
    test_set = ""
    output_results = ""
    
    if (len(sys.argv) > 3):
        model_name = sys.argv[1]
        test_set = sys.argv[2]
        output_results = sys.argv[3]
    
    sc = StandardScaler()
    # load the model from disk
    classifier = pickle.load(open('models/svc_classifier.sav', 'rb'))
    
    testset_x = pd.read_csv('data/test_set_x.csv')
    test_X = testset_x.iloc[:,1]
    test_features = extract_features(test_X.tolist())
    test_features = sc.fit_transform(test_features)
    y_test_results = classifier.predict(test_features)
            
    export_kaggle_results('kaggle_svc_91.csv', 'Id','Category', y_test_results)