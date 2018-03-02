import keras.callbacks
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy.random as random

LOOKBACK_WINDOW = 15
PREDICT_WINDOW = 1
BATCH_SIZE = 1     #e.g if 3, then trains with 3 samples/lookback windows, each with 15 timesteps and 2 features at once before updating gradients.
HIDDEN = 32
DROPOUT = 0.3
OPTIMIZER = 'adam'
EPOCHS = 3

#First, we create an artificial cointegrated time series.
def coint_path(N,delta,sigma,PO):
    X = [PO]
    Y = [PO]
    for i in range(N):
        dX = random.randn()+delta*(Y[-1] - X[-1])
        X.append(X[-1]+dX)
        dY = random.randn()+sigma*(X[-1] - Y[-1])
        Y.append(Y[-1]+dY)

    return X,Y

#create artificial cointegrated series
X,Y = coint_path(60,0.9,0.1,0)

#This is what the two series look like. Please note Y in this case does not mean it is a target value.
plt.plot(np.array(Y),label='Y')
plt.plot(np.array(X),label='X')
plt.legend()
plt.show();

# We need to ‘stack’ the two series together into a multidimensional array. 
#We then normalise the newly created multidimensional array using sklearn’s MinMaxScaler function. 
#This transforms the values to become between -1 and 1.
def prep_feature_data(X,Y):
    data = np.column_stack([X,Y])
    scaler = MinMaxScaler(feature_range=(-1,1))
    data = scaler.fit_transform(data)

return data

#format features for the model into a multidimensional array
data = prep_feature_data(X,Y)

#This is what the feature data looks like now, as a multidimensional array.
print(data[0:3])
print(data.shape)

def window_stop(data,LOOKBACK_WINDOW,PREDICT_WINDOW):
examples = LOOKBACK_WINDOW
y_examples = PREDICT_WINDOW
nb_samples = len(data) - examples - y_examples #makes sure it can be split into lookback windows properly

return nb_samples

#ensure it can be divided into the lookback window/batch size
nb_samples = window_stop(data,LOOKBACK_WINDOW,PREDICT_WINDOW)

print(nb_samples)


def input_features(nb_samples,LOOKBACK_WINDOW):
    input_list = [np.expand_dims(data[i:LOOKBACK_WINDOW+i,:], axis=0) for i in range(nb_samples)] #here nb_samples comes in handy
    input_mat = np.concatenate(input_list, axis=0)

return input_mat

#format the features into the batch size
input_mat = input_features(nb_samples,LOOKBACK_WINDOW)

print(input_mat[0:2]) #the second sample of features will become the first sample in the next window
print(input_mat.shape)


#Now, we create our target values. In this example, we want to train the LSTM model to predict the intercept of the linear regression line, 
#based on the slope (beta) between the two features.

def target_values(input_mat):
    targets = []
    for i in range(len(input_mat)):
        X1 = input_mat[i].T[0]
        X2 = input_mat[i].T[1]
        beta = np.polyfit(X1,X2,1)[0] #the slope is calculated from each lookback window 
        targets.append(X1[-1] - beta*X2[-1]) 
        targets = np.array(targets)

    return targets

#format the targets into the batch size
targets = target_values(input_mat)

#Typically before modelling we would split our feature and target data into training/test sets, 
#but because in reality we’d like to train on as much data we have available at a specific timestep, 
#we take the walk forward approach. This involves constantly training or refitting the model and predicting one step at a time.

def basic_LSTM_model(BATCH_SIZE,HIDDEN,LOOKBACK_WINDOW,DROPOUT,OPTIMIZER,EPOCHS,input_mat,targets):
    num_features = input_mat.shape[2] #this is the same as input dimension, used in describing batch_input_shape

    model = Sequential()

    model.add(LSTM(HIDDEN,batch_input_shape=(BATCH_SIZE,LOOKBACK_WINDOW,num_features)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1)) #the output is one dimensional

    model.compile(loss='mse',optimizer=OPTIMIZER)
    history = LossHistory()

    predictions,actuals,scores = [],[],[]

    assert(len(input_mat)==len(targets)) #checks each sample has a corresponding target value 

    for i in range(1,len(input_mat)):
    trainX = input_mat[0:i]
    trainY = targets[0:i]
    testX = input_mat[i].reshape(1,input_mat[0].shape[0],input_mat[0].shape[1]) #convert the input into 3 dimensional array
    testY = np.array([targets[i]]).reshape(1,1) #convert the target value into a 2 dimensional array

    #walk forward training, to predict the next timestep 
    #you can adjust the verbose parameter to 1 or 2 to watch the LSTM's progress 
    model.fit(trainX,trainY,nb_epoch=EPOCHS,batch_size=BATCH_SIZE,callbacks=[history],verbose=0)
    prediction = model.predict(testX,batch_size=BATCH_SIZE) 
    score = model.evaluate(testX,testY,batch_size=BATCH_SIZE,verbose=0)

    predictions.append(prediction[0][0])
    actuals.append(targets[i])
    scores.append(score)

    return model,history,predictions,actuals,scores


#We can evaluate the model by plotting the test score losses (these are based on mean squared error, as specified in model.compile), 
#the training losses over each batch (this has been saved using the keras callback in model.fit), and by visually comparing the actual and predicted values against each other.
def evaluate_walk_forward_LSTM(model,history,predictions,actuals,scores):
    print(np.mean(scores[int(len(scores)*0.75):])) #since the latter predictions have been trained on more data, we take the average of the testing loss scores for the last quarter of predictions

    plt.figure(1)
    plt.plot(history.losses)
    plt.title('Loss History')
    plt.figure(2)
    plt.plot(scores)
    plt.title('Testing Loss')
    plt.figure(3)
    plt.plot(actuals,'b-',label='actual')
    plt.plot(predictions,'g-',label='prediction')
    plt.title('Basic LSTM')
    plt.legend()
    plt.grid('on')
    plt.show()
    
    LOOKBACK_WINDOW = 15
    PREDICT_WINDOW = 1
    BATCH_SIZE = 1 #e.g if 3, then trains with 3 samples/lookback windows, each with 15 timesteps and 2 features at once.
    HIDDEN = 32
    DROPOUT = 0.3
    OPTIMIZER = 'adam'
    EPOCHS = 3
    
    model,history,predictions,actuals,scores = basic_LSTM_model(BATCH_SIZE,HIDDEN,LOOKBACK_WINDOW,DROPOUT,OPTIMIZER,EPOCHS,input_mat,targets)
    evaluate_walk_forward_LSTM(model,history,predictions,actuals,scores)
    
    
    
    
    
    
    
    
    
   
    
    
