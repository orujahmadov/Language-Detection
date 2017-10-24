# Language-Detection
Python implementation of Language Detection Algorithm for COMP 551 (Applied Machine Learning) class.

Authors:
Knight, Ryan, ryan.knight@mail.mcgill.ca, 260531961
Ahmadov, Oruj, oruj.ahmadov@mail.mcgill.ca, 260523568
Kalra, Anurag, anurag.kalra@mail.mcgill.ca, 260631195

Required Libraries: sklearn, keras, numpy, scipy, pandas, h5py
To install libraries: pip install sklearn keras numpy scipy pandas h5py

Following Machine Learning models used:

1) Linear: Logistic Regression
2) Non-linear: Decision Trees
3) Artificial Neural Networks

To train the first two model, run the following command:

python main.py train_set_x.csv train_set_y.csv model_name.sav

To train the neural networks, run the following command:

python main.py train_set_x.csv train_set_y.csv model_name.h5


To make prediction using first two model, run the following command:
python predictor model_name.sav output_file.csv

To make prediction using neural networks, run the following command:
python predictor model_name.h5 output_file.csv
