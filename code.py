######################### IMPORTING LIBRARIES:

import tensorflow as tf
from numpy import genfromtxt 
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#######################################################

################ONE HOT ENCODING:

def convertOneHot(data):
    y = np.array( [int(i[0]) for i in data]  )
    rows = len(y)
    columns = y.max() 
    a = np.zeros(shape=(rows, columns))
    for i,j in enumerate(y):
        a[i][j-1] = 1
    return (a)


#######################################################
###############LOADING DATA

data = genfromtxt('med_train1.csv',delimiter=',') 
test_data = genfromtxt('med_test1.csv',delimiter=',') 

A = data.shape[1]   #num features
B = data.shape[0]   #num classes
#samples_in_train = trainX.shape[0]
#samples_in_test = testX.shape[0]
#print A
#print B
#data = genfromtxt('train_test.txt', delimiter=',')
#test_data = genfromtxt('test.txt', delimiter=',')
x_train = np.array([i[1::] for i in data])
x_test = np.array( [i[1::] for i in test_data] )
#print x_test
y_train_onehot = convertOneHot(data)
y_test_onehot = convertOneHot(test_data)

#####################################################

#####################10 FOLD CROSS VALIDATION:

from sklearn.model_selection import KFold
def k_fold(x_train, y_train_onehot):
    kf = KFold(n_splits=10) 
    KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(x_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train_10fold, x_test_10fold = x_train[train_index], x_train[test_index]
        y_train_10fold, y_test_10fold = y_train_onehot[train_index], y_train_onehot[test_index]
        return x_train_10fold, x_test_10fold, y_train_10fold, y_test_10fold

#############################################

########CALLING FUNCTION FOR CROSS VALIDATION:
#########MAKING TRAIN AND VALIDATION SETS:
x_train_10fold, x_test_10fold, y_train_10fold, y_test_10fold = k_fold(x_train, y_train_onehot)

###################################################

###########FEATURE SCALING ON BOLTH ORIGNAL DATA SET AND K-FOLD TRAIN SET:
sc=StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)
x_test_10fold=sc.transform(x_test_10fold)
########SCaling
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train_10fold = scaler.fit_transform(x_train_10fold)

##############################################

#############FUNCTIONS FOR PRINTING F SCORES, RECALL & CONFUSION MATRIX:

precison_scores_list=[]

def print_stats_metrics(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred) )
    confmat= confusion_matrix(y_true=y_test, y_pred=y_pred)
    print confmat
    print pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Pred'], margins=True)
    precison_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred, average='weighted' ))
    print('precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted' ))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted' ))
    print('F1-Measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted' ))
#############################################################

############PLOTTING PRECISON PER EPOCH:

def plot_metric_per_epoch():
    x_epochs= []
    y_epochs= []
    for i,val in enumerate(precision_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    plt.scatter(x_epochs, y_epochs, s=50, c='lightgreen', marker='s', label='precision')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('score per epoch')
    plt.legend()
    plt.grid()
    plt.show()

######################################################

###############FUNCTION FOR COMPUTING Z VALUES, USING RANDOM INITIALIZATION FOR WEIGHTS AND RELU ACTIVATION FUNCTION

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


#######################################################

###########DEFINING THE ARCHITECHTURE OF MODEL(WITH 5 HIDDEN LAYERS)

def inference_deep(x, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [A, 280], [280] )
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [280,250], [250])
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [250,220], [220])
    with tf.variable_scope("hidden_4"):
        hidden_4 = layer(hidden_3, [220, 180], [180] )
    with tf.variable_scope("hidden_5"):
        hidden_5 = layer(hidden_4, [180, 150], [150] )
    with tf.variable_scope("output"):
        output = layer(hidden_5, [150, B], [B])
    return output


#########################################################
#############COST FUNCTION FOR MODEL:

def loss_deep(output, y):
    #output = tf.clip_by_value(output, 1e-10, 1.0)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01  # Choose an appropriate one.
    loss = loss + reg_constant * sum(reg_losses)
    return loss


#########################################################
############MODEL TRAINING & USING GRADIENT DESCENT FOR WEIGHT OPTIMIZATION:
WITH LEARNIG RATE (0.0001)

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train_op = optimizer.minimize(cost)
    return train_op

##########################################################

#############EVALUATIONG THE PERFORMANCE OF MODEL:

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy



###########################################################
###########DEFINING PLACEHOLDERS AND CALLING FUNCTIONS:

x = tf.placeholder("float", [None, 300]) # 4 features
y = tf.placeholder("float", [None, 3]) # 3 classes

output = inference_deep(x, 300, 3)  ## 300 FEATURES AND 3 CLASSES
cost = loss_deep(output, y)

train_op = training(cost)
eval_op = evaluate(output, y)

#####################################

y_p_metrics=tf.argmax(output, 1)  # to get the index

##########################################
##############INITIALIZE ALL VARIABLES

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

##########################################
###############BATCHING TO DIVIDE DATA INTO CHUNKS AND PROCESSING CHUNK BY CHUNK

batch_size = 100
Folds=10
n_epochs = 1000  ####### NUMBER OF TRAINING SAMPLES
num_samples_train_set = x_train_10fold.shape[0]
num_batches = int(num_samples_train_set/batch_size)

#####################################

for i in range(n_epochs):  ## FOR TRAINING SAMPLES
    print("iteration %d " % i)
    for i in  range(Folds):  ## FOR K FOLDS
        x_train_10fold, x_test_10fold, y_train_10fold, y_test_10fold = k_fold(x_train, y_train_onehot)        
        for batch_n in range(num_batches): ## FOR BATCHING
            
            sta = batch_n*batch_size
            end = sta + batch_size
            sess.run(train_op, feed_dict={x: x_train_10fold[sta:end,:], y: y_train_10fold[sta:end,:]})### TRAINING FOR 10 FOLD 
        result=sess.run(eval_op, feed_dict={x: x_test_10fold, y: y_test_10fold })
        print "Training{}, {}".format(i, result)   ### VALIDATION FOR 10 FOLD
    print "************************************************************
    result1, y_result_metrics = sess.run([eval_op, y_p_metrics], feed_dict={x: x_test, y: y_test_onehot} )  #TESTING FOR MODEL
    y_true = np.argmax(y_test_onehot,1)
    print_stats_metrics(y_true, y_result_metrics ) ### PRINTING STATS METRICES
    print "Run{}, {}".format(i, result1)
    plot_metric_per_epoch ## printing epochs with precision scores 

##########################################

print "<<<<<<<DONE>>>>>>>>"
