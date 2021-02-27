import warnings
import itertools
from math import sqrt
from datetime import datetime
from numpy import concatenate
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
from matplotlib import pyplot
from entsoe import EntsoePandasClient


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lower_confidence_interval(maindata, prediction):
    s = np.std(maindata) # std of vector
    z = 1.96 # for a 95% CI
    lower = prediction - (z * s)
    return lower

def upper_confidence_interval(maindata, prediction):
    s = np.std(maindata) # std of vector
    z = 1.96 # for a 95% CI
    upper = prediction + (z * s)
    return upper


#%%%###API#############
def EPIAS_API():
    down = './test.json'
    url = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate=2019-12-31&startDate=2017-01-01'
    outpath=down
    generatedURL=url
    response = requests.get(generatedURL)
    if response.status_code == 200:
        with open(outpath, "wb") as out:
            for chunk in response.iter_content(chunk_size=128):
                out.write(chunk)
    with open(down) as json_file:
        data = json.load(json_file)
    body=data.get('body')
    gen=body.get('dayAheadMCPList')
    df=pd.DataFrame(gen)
    return(df)

#%%#############ENTSOE-API####################
def ENTSOE_API():
    client = EntsoePandasClient(api_key="2c958a88-3776-4f01-82cd-c957fdc4dc6a")

    country_code = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE'

    start = [pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2017-01-01T00:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z')]
    end= [pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T22:00Z')]

    df1=[]
    iteration2=0
    ElectricityPrice=[]
    for iiii in range(len(country_code)):
        ElectricityPrice=client.query_day_ahead_prices(country_code[iteration2], start=start[iteration2], end=end[iteration2])
        if iiii==0:
            df1=pd.DataFrame({country_code[iteration2]:ElectricityPrice.values})
            iteration2=iteration2+1
            print(df1)
        else:
            df1[country_code[iteration2]]=pd.DataFrame({country_code[iteration2]:ElectricityPrice.values})
            iteration2=iteration2+1
            print(df1)
        # print(len(ElectricityPrice))
    return(df1)
#%%############SPLIT TRAIN TEST MODEL#############
df = ENTSOE_API()
df1 = EPIAS_API()
df1 = df1['priceEur']

countrycode = 'TR'
df[countrycode]=pd.DataFrame({countrycode:df1.values})

labels = df[26112:26280]
df = df[0:26112]
#%%####################################################################
newcountrycode = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE', 'TR'
countriesnames = 'Estonia', 'Portugal', 'Spain', 'France', 'Finland', 'Hungary', 'Slovenia', 'Latvia', 'Netherlands', 'Greece', 'Belgium', 'Turkey'
fig, ax = plt.subplots(3, 4, figsize=(25,15))
RMSEresult = list()
iter1=0
iteration4=0
iteration5=0
TahminLSTM = pd.DataFrame(columns=df.columns).fillna(0)
lowerLSTM = pd.DataFrame(columns=df.columns).fillna(0)
upperLSTM = pd.DataFrame(columns=df.columns).fillna(0)
statisticsLSTM = pd.DataFrame(columns=df.columns).fillna(0)

for iteration in range(len(newcountrycode)):
    
    values = df[newcountrycode[iteration]].values.reshape(-1,1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(values)
    # Birkaç Değere Bakalım
    dataset[0:5]
    
    # %80 Train % 20 Test
    TRAIN_SIZE = 0.80
    train_size = int(len(dataset) * TRAIN_SIZE)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("Gün Sayıları (training set, test set): " + str((len(train), len(test))))

    def create_dataset(dataset, window_size = 48):
        data_X, data_Y = [], []
        for i in range(len(dataset) - window_size - 1):
            a = dataset[i:(i + window_size), 0]
            data_X.append(a)
            data_Y.append(dataset[i + window_size, 0])
        return(np.array(data_X), np.array(data_Y))


    window_size = 48
    train_X, train_Y = create_dataset(train, window_size)
    test_X, test_Y = create_dataset(test, window_size)
    print("Original training data shape:")
    print(train_X.shape)
    # Yeni verisetinin şekline bakalım.
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    print("New training data shape:")
    print(train_X.shape)
#%%###############LSTM MODEL#####################
    def fit_model(train_X, train_Y, window_size = 48):
        model = Sequential()
    
        # Modelin tek layerlı şekilde kurulacak.
        model.add(LSTM(5, input_shape = (1, window_size)))
        model.add(Dense(1))
        model.compile(loss = 'mae', optimizer = 'adam')
    
        #30 epoch yani 30 kere verisetine bakılacak.
        model.fit(train_X, train_Y, epochs = 15, batch_size = 50, verbose = 2, validation_data=(test_X, test_Y), shuffle=False)
        # plot train and validation loss
        # pyplot.plot(history.history['loss'])
        # pyplot.plot(history.history['val_loss'])
        # pyplot.title('model train vs validation loss')
        # pyplot.ylabel('loss')
        # pyplot.xlabel('epoch')
        # pyplot.legend(['train', 'validation'], loc='upper right')
        # pyplot.show()
        return(model)
    # Fit the first model.
    model1 = fit_model(train_X, train_Y, window_size)

#%%#####################PREDICTION#########################
    def predict_and_score(model, X, Y):
        # Şimdi tahminleri 0-1 ile scale edilmiş halinden geri çeviriyoruz.
        pred = scaler.inverse_transform(model.predict(X))
        orig_data = scaler.inverse_transform([Y])
        # Rmse değerlerini ölçüyoruz.
        score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
        return(score, pred)

    rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
    rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
    print("Training data score: %.2f RMSE" % rmse_train)
    print("Test data score: %.2f RMSE" % rmse_test)

#%%################PLOT####################
    # Öğrendiklerinini tahminletip ekliyoruz.
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict
    # Şimdi ise testleri tahminletiyoruz.
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict
    
    
    test_predict_plot = pd.DataFrame(test_predict_plot)
    TahminLSTM[newcountrycode[iteration]] = test_predict_plot[0][25943:26111]
    lowerLSTM[newcountrycode[iteration]] = lower_confidence_interval(df[newcountrycode[iteration]], TahminLSTM[newcountrycode[iteration]])
    upperLSTM[newcountrycode[iteration]] = upper_confidence_interval(df[newcountrycode[iteration]], TahminLSTM[newcountrycode[iteration]])
################
# RMSEresult = rmse(labels.loc[26113:26280].values, test_predict_plot[25944:26111])
# print(RMSEresult)
# plt.figure(figsize = (15, 5))
# plt.plot(labels.loc[26113:26280].values, label = "Actual Price (last 168 Hours)")
# plt.plot(test_predict_plot[25944:26111], label = "Test set prediction (last 168 Hours)")
# plt.xlabel("Hours")
# plt.ylabel("Price (Euro/MWh)")
# plt.title("Long Short-Term Memory (Real and Predicted Prices)")
# plt.legend()
# plt.savefig('plot.png', dpi=300)
# plt.show()

    # RMSEresult.append(rmse(labels.loc[26113:26280].values, test_predict_plot[25944:26111]))
    x = np.arange(0, 168)
    ax[iteration4, iteration5].plot(x, test_predict_plot[0][25943:26111], '.-', linewidth=3, color='tab:blue', label = "Test set prediction (last 168 Hours)")
    ax[iteration4, iteration5].plot(x, labels[newcountrycode[iteration]].loc[26112:26280], '.-', linewidth=3, color='tab:orange', label="Actual Price (last 168 Hours)")
    ax[iteration4, iteration5].fill_between(x, lowerLSTM[newcountrycode[iteration]], upperLSTM[newcountrycode[iteration]], color='dodgerblue', alpha=0.2, label="Confidence Interval (95%)")
    ax[iteration4, iteration5].set_title(countriesnames[iter1], fontsize=20)
    ax[iteration4, iteration5].set_xlabel('Hours', fontsize=12)
    ax[iteration4, iteration5].set_ylabel('Prices (Euro/MWh)', fontsize=12)
    ax[iteration4, iteration5].legend(loc="best")
    ax[iteration4, iteration5].legend(fontsize=8)
    iter1 = iter1 + 1
    iteration5 = iteration5 + 1
    if iteration5 == 4:
        iteration4 = iteration4 + 1
        iteration5 = 0
fig.tight_layout()
# plt.savefig('LSTM_48.png')
# plt.savefig('LSTM_48.eps')
plt.show()




