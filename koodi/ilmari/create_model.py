import pandas as pd
import statsmodels.api as sm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from handle import boxcox_df

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
    if boxcox:
        cols = list(df.columns)
        cols.remove("quality")
        df,trans = boxcox_df(df,cols=cols)
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
    pred_count = Counter(preds.to_numpy().flatten())
    obs_count = Counter(y_test.to_numpy().flatten())
    print("Neural network predictions",pred_count.items())
    print("Actual values",obs_count.items())
    count = 0
    for obs, pred in zip(y_test,preds):
        if int(obs) == int(pred):
            count += 1
    print("Accuracy of model: ", round(count/len(preds),3))
    errs = preds - y_test
    fig,ax = plt.subplots()
    bins = list(range(9))
    ax.bar(list(obs_count.keys()),list(obs_count.values()),label="Observations")
    ax.bar(pred_count.keys(),pred_count.values(),label="Predictions")
    #print(errs)
    #ax.scatter(y_test,errs)
    #ax.set_title(f"Residuals")
    ax.legend()
    fig,ax = plt.subplots()
    ax.hist(errs)
    plt.show()
    
def multip_rows(df,ntimes=3,mask_cond=None):
    if mask_cond is None:
        mask_cond = lambda df : (np.abs(scipy.stats.zscore(df)) >= 3).any(axis=1)
    weirds = df[mask_cond(df)]
    print(f"Found {len(weirds)} weird rows.")
    weirds = pd.concat([weirds.copy() for _ in range(ntimes)])
    df = pd.concat([df,weirds])
    print(f"Added {len(weirds)} rows.")
    return df
    
def add_rows_red_wine_regress(x_train, y_train):
    # multiple values in train set
    train = pd.concat([x_train,y_train],axis=1)
    print(train)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3/8]),ntimes=8)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([4/8]),ntimes=2)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([1]),ntimes=7)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([7/8]),ntimes=1)
    y_train = train.pop("quality")
    x_train = train
    return x_train, y_train

def red_wine_model_regress(xy_splits,train=True):
    # Looks like we shouldn't get rid of the outliers, when creating a neural network.
    # They likely help the neural network to generalize
    x_train,x_test, y_train,y_test = xy_splits
    x_train, y_train = add_rows_red_wine_regress(x_train, y_train)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = tf.keras.models.Sequential(
    layers=[
        tf.keras.layers.Dense(30,activation="linear"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(8,activation="elu"),
        tf.keras.layers.Dense(256,activation="linear"),
        tf.keras.layers.Dense(1,"sigmoid"),
        ]
    )
    model.build(np.shape(x_train))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, amsgrad=True),
        loss=tf.keras.losses.LogCosh(),
        )
    if train:
        model.build(np.shape(x_train))
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, amsgrad=True),
        loss=tf.keras.losses.LogCosh(),
        )
        print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        hist = model.fit(
            x_train, y_train,
            epochs=20,
            verbose=1,
            validation_data=(x_test,y_test),
            batch_size=128,
        )
    else:
        model.load_weights("./koodi/ilmari/red-wine-regress-model.h5")
    return model

def red_wine_forest_regress(exog,endog):
    dec_tree = RandomForestRegressor(n_estimators=32,random_state=42,verbose=0,n_jobs=8)
    dec_tree.fit(np.array(exog),np.array(endog))
    return dec_tree

    
if __name__ == "__main__":
    df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
    dec_tree = RandomForestRegressor(n_estimators=32,random_state=42,verbose=0,n_jobs=8)
    xy_split = get_winedata_split(df,rm_outliers=False)

    dec_tree.fit(np.array(xy_split[0]),np.array(xy_split[2]))
    test_model(dec_tree, np.array(xy_split[1]),np.array(xy_split[3]))
    #xy_split = get_winedata_split(df)
    #rw_model = red_wine_model(xy_split,train=True)
    #test_model(rw_model,np.array(xy_split[1]),np.array(xy_split[3]))
    
    