import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix;
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)

# load the dataset
time = read_csv('data.csv', usecols=[0], engine='python')
buy_sell = read_csv('data.csv', usecols=[1], engine='python')
transactions = read_csv('data.csv', usecols=[2], engine='python')

# transform dataframe to numpy array
time = time.values
buy_sell = buy_sell.values
transactions = transactions.values

#dataset = dataframe.values
#dataset = dataset.astype('float32')
# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)

# compute the sum of buy/sell transactions of each day
num_datapoints = 24 * 12  # number of transactions per day
num_days = 31 + 28 + 31 + 30 + 31 + 30 + 23  #  number of days in 2017

# create the time series of sum of buy/sell transactions of each day
buys = []
sells = []
diff = []
for i in range(num_days):
    buy = 0
    sell = 0
    start = i * num_datapoints
    for j in range( num_datapoints ):
        if math.isnan( transactions[ start + j, 0 ] ):
            print( "nan is ", start + j )
            continue
        elif  buy_sell[ start + j ] == "SE":
            sell = sell + transactions[ start + j, 0 ]
            sells.append(sell)
        else:
            buy = buy + transactions[ start + j, 0 ]
            buys.append(buy)
    diff.append( buy - sell )


# split the diff into train and test sets
# train_size = int(len(diff) * 0.67)
# test_size = len(diff) - train_size
# train, test = diff[0:train_size,:], diff[train_size:len(diff),:]

cv_params = numpy.arange(0.7)

for cv_param in cv_params:
    train_size = int(len(diff) * cv_param)
    test_size = len(diff) - train_size
    train, test = diff[0:train_size], diff[train_size:len(diff)]
    # fit the model
    look_back = 10
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    numpy.array(diff)
    numpy.array(sells)
    numpy.array(buys)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict



# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
