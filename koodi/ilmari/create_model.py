import pandas as pd
import statsmodels.api as sm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import tensorflow as tf
from sklearn.model_selection import train_test_split

from handle import max_norm
def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data,dtype=int)
    arr = np.zeros((len(targets),nb_classes),dtype=int)
    print(targets)
    for i,_ in enumerate(arr):
        arr[i,int(targets[i])] = 1
    return arr

def get_winedata_split(df,rm_outliers=False):
    # Read and handle data
    if rm_outliers:
        df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
    endog = df.pop("quality")
    endog = pd.DataFrame(indices_to_one_hot(endog,10))
    exog = max_norm(df)
    return train_test_split(exog,endog,test_size=0.25,random_state=42)

def test_model(model,x_test,y_test):
    preds = pd.DataFrame(model.predict(x_test))
    preds = preds.idxmax(axis=1)
    y_test = pd.DataFrame(y_test)
    y_test = y_test.idxmax(axis=1)
    print(preds)
    print(y_test)
    print("Neural network pedictions",Counter(preds).items())
    print("Actual values",Counter(y_test).items())
    count = 0
    for obs, pred in zip(y_test,preds):
        if obs == pred:
            count += 1
    print("Accuracy of model: ", round(count/len(preds),3))
    errs = preds - y_test
    fig,ax = plt.subplots()
    #print(errs)
    #ax.scatter(y_test,errs)
    ax.hist(errs)
    plt.show()

def red_wine_model(xy_splits,train=True):
    # Looks like we shouldn't get rid of the outliers, when creating a neural network.
    # They likely help the neural network to generalize
    x_train,x_test, y_train,y_test = xy_splits
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    """
    layers=[
        tf.keras.layers.Dense(3,activation="linear"),
        tf.keras.layers.Dense(8,activation="linear"),
        tf.keras.layers.Dropout(0.00131),
        tf.keras.layers.Dense(10,activation="linear"),
        tf.keras.layers.Dropout(0.01947),
        tf.keras.layers.Dense(12,activation="linear"),
        tf.keras.layers.Dense(10,"softmax"),
        ]
    """
    model = tf.keras.models.Sequential(
    layers=[
        tf.keras.layers.Dense(512,activation="linear"),
        tf.keras.layers.Dense(85,activation="linear"),
        tf.keras.layers.Dropout(0.00131),
        tf.keras.layers.Dense(4,activation="linear"),
        tf.keras.layers.Dropout(0.01147),
        tf.keras.layers.Dense(12,activation="linear"),
        tf.keras.layers.Dense(10,"softmax"),
        ]
    )
    model.build(np.shape(x_train))
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.025375482208110867, decay=0.14532385839041545,momentum=0,rho=0,epsilon=10**-7),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
    if train:
        model.build(np.shape(x_train))
        model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.025375482208110867, decay=0.14532385839041545,momentum=0,rho=0,epsilon=10**-7),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
        print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        hist = model.fit(
            x_train, y_train,
            epochs=100,
            verbose=1,
            validation_data=(x_test,y_test),
            batch_size=256,
        )
    else:
        model.load_weights("./koodi/ilmari/red-wine-model.h5")
    return model
    
if __name__ == "__main__":
    df = pd.read_csv("./viinidata/winequality-white.csv",sep=";")
    xy_split = get_winedata_split(df)
    rw_model = red_wine_model(xy_split,train=True)
    test_model(rw_model,np.array(xy_split[1]),np.array(xy_split[3]))
    
    