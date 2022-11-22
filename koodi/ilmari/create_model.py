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

def get_winedata_split(df,rm_outliers=False,boxcox=False,norm=False,add_rares = True,onehot=True,wine="red"):
    # Read and handle data
    if rm_outliers:
        df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
    endog = df.pop("quality")
    exog = df
    if norm:
        exog = max_norm(exog)
        if not onehot:
            endog = max_norm(pd.DataFrame(endog))
            
    if boxcox:
        df = pd.concat([exog,endog],axis=1)
        cols = list(df.columns)
        cols.remove("quality")
        df,trans = boxcox_df(df,cols=cols)
        endog = df.pop("quality")
        exog = df
    x_train,x_test,y_train,y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    if add_rares:
        assert not boxcox, "Cannot add rares if the data is transformed with box cox"
        if wine == "red":
            x_train,y_train = add_rows_red_wine_regress(x_train, y_train,normed=False)
        elif wine == "white":
            x_train,y_train = add_rows_red_wine_regress(x_train, y_train,normed=False)
    if onehot:
        y_train = pd.DataFrame(indices_to_one_hot(y_train,10))
        y_test = pd.DataFrame(indices_to_one_hot(y_test,10))
    return x_train,x_test,y_train,y_test

def test_model(model,x_test,y_test,wine="red"):
    preds = pd.DataFrame(model.predict(x_test))
    y_test = pd.DataFrame(y_test)
    if len(np.shape(np.array(y_test))) > 1 and 1 not in np.shape(np.array(y_test)):
        preds = preds.idxmax(axis=1)
        y_test = y_test.idxmax(axis=1)
    else:
        mul = 8 if wine=="red" else 9
        preds = pd.Series(preds.apply(lambda x : round(mul*x)).to_numpy().flatten())
        y_test = pd.Series(y_test.apply(lambda x : round(mul*x)).to_numpy().flatten())
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
    ax.bar(list(obs_count.keys()),list(obs_count.values()),label="Observations",alpha=0.75)
    ax.bar(pred_count.keys(),pred_count.values(),label="Predictions",alpha=0.75)
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

def add_rows_white_wine_regress(x_train, y_train, normed=True):
    # multiple values in train set
    train = pd.concat([x_train,y_train],axis=1)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3/9 if normed else 3]),ntimes=6) #13
    #train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([7/9]),ntimes=4)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([1 if normed else 9]),ntimes=10)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([8/9 if normed else 8]),ntimes=2)
    y_train = train.pop("quality")
    x_train = train
    return x_train, y_train
    
    
def add_rows_red_wine_regress(x_train, y_train,normed=True):
    # multiple values in train set
    train = pd.concat([x_train,y_train],axis=1)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3/8 if normed else 3]),ntimes=8)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([4/8 if normed else 4]),ntimes=2)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([1 if normed else 8]),ntimes=7)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([7/8 if normed else 7]),ntimes=1)
    y_train = train.pop("quality")
    x_train = train
    return x_train, y_train

def red_wine_model_regress(xy_splits,train=True):
    # Looks like we shouldn't get rid of the outliers, when creating a neural network.
    # They likely help the neural network to generalize
    x_train,x_test, y_train,y_test = xy_splits
    if len(np.shape(np.array(y_test))) > 1 and 1 not in np.shape(np.array(y_test)):
        print("Turning one hot vectors to normal vectors:", np.shape(np.array(y_test)))
        y_test = y_test.idxmax(axis=1)
        y_train = y_train.idxmax(axis=1)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = tf.keras.models.Sequential(
        [
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
            verbose=2,
            validation_data=(x_test,y_test),
            batch_size=128,
        )
    else:
        model.load_weights("./koodi/ilmari/red-wine-regress-model.h5")
    return model

def white_wine_model_regress(xy_splits,train=True):
    # Looks like we shouldn't get rid of the outliers, when creating a neural network.
    # They likely help the neural network to generalize
    x_train,x_test, y_train,y_test = xy_splits
    if len(np.shape(np.array(y_test))) > 1 and 1 not in np.shape(np.array(y_test)):
        print("Turning one hot vectors to normal vectors:", np.shape(np.array(y_test)))
        y_test = y_test.idxmax(axis=1)
        y_train = y_train.idxmax(axis=1)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model = tf.keras.models.Sequential(
        [
        tf.keras.layers.Dense(49,activation="elu"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(8,activation="softmax"),
        tf.keras.layers.Dense(256,activation="linear"),
        tf.keras.layers.Dense(1,"sigmoid"),
        ]
    )
    model.build(np.shape(x_train))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, amsgrad=True),
        loss=tf.keras.losses.MeanAbsoluteError(),
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
            verbose=2,
            validation_data=(x_test,y_test),
            batch_size=256,
        )
    else:
        model.load_weights("./koodi/ilmari/white-wine-regress-model.h5")
    return model

def red_wine_forest_regress(exog,endog):
    dec_tree = RandomForestRegressor(n_estimators=32,random_state=42,verbose=0,n_jobs=8,)
    dec_tree.fit(np.array(exog),np.array(endog))
    return dec_tree

def white_wine_forest_regress(exog,endog):
    dec_tree = RandomForestRegressor(n_estimators=64,random_state=42,verbose=0,n_jobs=8,)
    dec_tree.fit(np.array(exog),np.array(endog))
    return dec_tree

def test_whitewine_regress_model():
    # Test on full dataset
    df = pd.read_csv("./viinidata/winequality-white.csv",sep=";")
    xy_split = get_winedata_split(df,boxcox=False,rm_outliers=False,norm=True,add_rares=True,onehot=False,wine="white")
    print(xy_split)
    #dec_tree = white_wine_forest_regress(np.array(xy_split[0]),np.array(xy_split[2]))
    dec_tree = white_wine_model_regress(xy_split,train=True)
    test_model(dec_tree, np.array(xy_split[1]),np.array(xy_split[3]),wine="white")

def test_redwine_regress_model():
    # Test on full dataset
    df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
    xy_split = get_winedata_split(df,boxcox=False,rm_outliers=False,norm=True,add_rares=True,onehot=False)
    print(xy_split)
    #dec_tree = red_wine_forest_regress(np.array(xy_split[0]),np.array(xy_split[2]))
    dec_tree = red_wine_model_regress(xy_split,train=False)
    test_model(dec_tree, np.array(xy_split[1]),np.array(xy_split[3]),wine="red")

    
if __name__ == "__main__":
    test_whitewine_regress_model()
    
    