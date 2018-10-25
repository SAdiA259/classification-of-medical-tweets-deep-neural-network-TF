# classification-of-medical-tweets-deep-neural-network-TF
classifying medical tweets with TF deep neural network applying logistic regression

Classification of personal medication intake

OBJECTIVE: •	To read tweets from twitter and save it in a text file along with classes. •	To read data from text file and perform vectorization. •	To perform logistic regression deep neural network in TF and compare results with actual data with the help of accuracy scores and confusion matrix

Procedure:

I used the file "tweets" to read data from twitter using tweed ids.
I used the code “get_twitter” to get data from the "tweets" file.
Then I saved the output in data.text which contains tweets along with their id and classes.
I read through the file “data.txt” line by line , and its columns are separated by tab, I split every line into tabs and stored classes which are at index 3 in “list_of_classes” and tweets which are at index 4 in “list_of_tweets”
After performing SVD and saved it in file “lsafinal.txt”
I used train_test_split to to split data into training and test sets.
Then I used “Count Vectorizer” to covert tweets into features and then I used “Truncated Svd” for decomposing that features into 300.
Then I made two csv’s ‘med_train1.csv” for train set and “med_test1.csv” for test set.
I am Joining the vectorized data along with their classes and print it in the two csv, And there separation is “,”.
I am implementing Logistic Regression in tensor flow and testing the accuracy of model.
I am using K_fold_cross_validation technique with K =10  to randomly split training set into k folds without replacement,(k-1)folds for model training & 1 fold for performance evaluation.
I have used f scores, confusion matrix and accuracy score to test the performance of Model.
I have used One hot Encoding to convert categorical data to integer data by converting one column Y matrix into 3 column matrix because there are threeclasses.
I used SOFTMAX function so that the z values calculated by each layer neural network is equal to 1.
I have used Rectified Linear Unit (Relu) as an activation function.
I have initialized all weights randomly for better model prediction.
Number of epochs =1000, learning rate=0.0001 and layers=5
