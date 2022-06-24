# Libraries

import matplotlib.pyplot as plt                                                             # useful for visualizing data
import matplotlib.image as mpimg
plt.style.use('dark_background')

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import keras.utils
import tensorflow as tf                                                                     # A set of tools for machine learning
from tensorflow import keras
from sklearn.model_selection import train_test_split                                        # a tool for separating sample data into training and testing data

from tensorflow.keras.layers import Flatten                                                 # tools for working with data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense, BatchNormalization,Dropout
from tensorflow.keras import optimizers

import numpy as np                                                                          # some math tools
import pandas as pd
import seaborn as sns 

from tensorflow.keras.optimizers import SGD 

from sklearn.linear_model import LogisticRegression                                         # We will be using logistic regression to find confidence intervals
from keras.wrappers.scikit_learn import KerasClassifier                                     # some machine learning tools
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

seed = 7                                                                                    # the 'random seed' seeds the algorithm
np.random.seed(seed)
tf.random.set_seed(13)
tf.debugging.set_log_device_placement(False)

                                                                                            # Here is the sample data, collected manually from the device
Flat = np.array([[903,1194,928,1004,0,959],                                                 
                [923,1239,945,1018,923,977],
                [927,1172,955,1024,0,959],
                [928,1422,952,1022,0,956],
                [924,1214,945,1019,923,976]])
Flat = Flat - np.average(Flat)                                                              # This describes a flat plane in the field of view (FoV)
cuLF = np.array([[729,3466,840,926,542,599],                                                
                [715,3475,965,983,630,648],
                [701,3473,809,959,595,659],
                [732,21111,812,941,594,687],
                [736,3476,844,988,605,670]])
cuLF = cuLF - np.average(cuLF)                                                              # This describes a curved surface to the left of the FoV
cuDF = np.array([[834,643,886,884,815,868],
                [906,688,883,902,808,889],
                [753,688,794,857,842,874],
                [749,688,819,884,867,872],
                [690,678,847,885,824,874]])
cuDF = cuDF - np.average(cuDF)                                                              # This describes a curved surface at the bottom of the FoV
cuRF = np.array([[586,913,634,873,0,836],
                [621,992,686,914,0,880],
                [659,951,641,902,0,846],
                [572,3354,602,875,0,843],
                [593,3355,624,891,0,857]])
cuRF = cuRF - np.average(cuRF)                                                              # This describes a curved surface to the right of the FoV
cuUF = np.array([[461,3376,470,699,0,924],
                [412,3456,452,636,407,686],
                [419,3372,456,561,0,662],
                [422,3380,456,585,0,687],
                [450,3384,438,520,0,617]])
cuUF = cuUF - np.average(cuUF)                                                              # This describes a surface that slopes at the top of the FoV

labels = np.array([[0],                                                                     # Here are the 'labels' for the above data
[0],
[0],
[0],
[0],
[1],
[1],
[1],
[1],
[1],
[2],
[2],
[2],
[2],
[2],
[3],
[3],
[3],
[3],
[3],
[4],
[4],
[4],
[4],
[4]])

feature_set = np.vstack([Flat,cuLF,cuDF,cuRF,cuUF])                                         # Here we organize the data and labels
dataSet = np.append(feature_set,labels,axis=1)
dataSet = pd.DataFrame(dataSet, columns = ['Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','class'])
# print(feature_set)
# print(labels)
# print(dataSet)
all_ds = dataSet.sample(frac=1)
#print(all_ds)


train_dataset, temp_test_dataset = train_test_split(all_ds,test_size=0.5)                   # Here we split the sample data into training and testing sets, half and half
#print(train_dataset.shape)
#print(temp_test_dataset.shape) 
test_dataset, valid_dataset = train_test_split(temp_test_dataset,test_size = 0.5)
# some print statements can go here to show the split

train_stats = train_dataset.describe()                                                      # Here we initialize train_... and test_... to evaluate our data  
train_stats.pop("class")
sns.pairplot(train_stats[train_stats.columns], diag_kind="kde")
train_stats = train_dataset.describe()
train_stats.pop("class")
train_stats=train_stats.transpose()
#print(train_stats)
train_labels1 = train_dataset.pop('class')
test_labels1 = test_dataset.pop('class')
valid_labels1 = valid_dataset.pop('class')
train_labels = pd.get_dummies(train_labels1,prefix='class')
test_labels = pd.get_dummies(test_labels1,prefix='class')
valid_labels = pd.get_dummies(valid_labels1,prefix='class')
#print(train_labels)

def norm(x):
    return (x - train_stats['mean'])/train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_valid_data = norm(valid_dataset)

x_train = normed_train_data
x_test = normed_test_data
y_train = train_labels1
y_test = test_labels1

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dummy_Y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(5, input_dim=6, kernel_initializer='normal', activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(Dense(5, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(predictions)