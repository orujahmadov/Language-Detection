# Language-Detection
Python implementation of Language Detection Algorithm for COMP 551 (Applied Machine Learning) class.

Authors:
- Knight, Ryan, ryan.knight@mail.mcgill.ca, 260531961
- Ahmadov, Oruj, oruj.ahmadov@mail.mcgill.ca, 260523568
- Kalra, Anurag, anurag.kalra@mail.mcgill.ca, 260631195

Required Libraries: sklearn, keras, numpy, scipy, pandas, h5py

To install libraries: pip install sklearn keras numpy scipy pandas h5py

Following Machine Learning models used:

1) Linear: Logistic Regression
2) Non-linear: Decision Trees
3) Artificial Neural Networks

There are 3 main components. Preprocessing, Training and Prediction.

To preproces data, run the following command:

python preprocess.py   path/../train_set_x.csv   path/../train_set_y.csv   path/../preprocessed.csv

path/../train_set_x.csv is path to train X csv file.

path/../train_set_y.csv is path to train Y csv file.

path/../preprocessed.csv is path to save preprocessed file.


To train the first two model, run the following command:

python main.py   path/../preprocessed.csv   path/../model_name.sav

Where:

path/../preprocessed.csv is path to preprocessed file.

path/../model_name.sav is path to save model to.


To train the neural networks, run the following command:

python main.py   path/../preprocessed.csv   path/../model_name.h5

path/../preprocessed.csv is path to preprocessed file.

path/../model_name.h5 is path to save model to.


To make prediction using first two model, run the following command:

python predictor   path/../model_name.sav   path/../output_file.csv

path/../model_name.sav is path to load model from.

path/../output_file.csv is path to save output results to.


To make prediction using neural networks, run the following command:

python predictor   path/../model_name.h5   path/../output_file.csv

path/../model_name.h5 is path to load model from.

path/../output_file.csv is path to save output results to.
