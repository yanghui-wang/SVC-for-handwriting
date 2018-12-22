This project trains a Support Vector Classifier to read handwriting. 

The startup code separate.py converts a column (rotated above to fit on page) of single characters into a list of separate images using a very simple algorithm. The code also resizes each of the letters to a 10 × 10 square.  
Generate your own handwriting data: using a dark pen on a white sheet of paper, write in a column several examples of a single symbol. Scan or photograph (cell phone cameras can work) your data and separate it into separate images for each symbol. Generate data and target arrays for your handwriting data. Five examples of each symbol should be sufficient if your handwriting is consistent.  
Once you have arrays of all your own handwriting data and target values, the function partition accepts your data and target arrays and a third parameter p representing the percentage of the data that is to be used for training; the remainder will be for testing. It should return train data, train target, test data, and test target.  
Train and test a Support Vector Classifier (SVC).
Print results like:      	Predicted: [ 2. 0. 1. 1. 2. 2.]
	Truth: [2. 0. 1. 1. 2. 2.]
	Accuracy:  100.0 %

