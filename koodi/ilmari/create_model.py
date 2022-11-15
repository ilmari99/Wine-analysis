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

def multip_rows(df,ntimes=3,mask_cond=None):
    if mask_cond is None:
        mask_cond = lambda df : (np.abs(scipy.stats.zscore(df)) >= 3).any(axis=1)
    weirds = df[mask_cond(df)]
    print(f"Found {len(weirds)} weird rows.")
    weirds = pd.concat([weirds.copy() for _ in range(ntimes)])
    df = pd.concat([df,weirds])
    print(f"Added {len(weirds)} rows.")
    return df

def get_winedata_split(df,rm_outliers=False,boxcox=False,norm=False,add_rares = True):
    # Read and handle data
    if rm_outliers:
        df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
    if boxcox:
        cols = list(df.columns)
        cols.remove("quality")
        df,trans = boxcox_df(df,cols=cols)
    endog = df.pop("quality")
    exog = df
    if norm:
        exog = max_norm(df)
    x_train,x_test,y_train,y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    if add_rares:
        train = pd.concat([x_train,y_train],axis=1)
        train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3/8 if norm else 3]),ntimes=6)
        train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([4/8 if norm else 4]),ntimes=1)
        train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([1 if norm else 8]),ntimes=5)
        y_train = train.pop("quality")
        x_train = train
    
    y_train = pd.DataFrame(indices_to_one_hot(y_train,10))
    y_test = pd.DataFrame(indices_to_one_hot(y_test,10))
    return x_train,x_test,y_train,y_test

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
    df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
    dec_tree = RandomForestRegressor(n_estimators=10,random_state=42,verbose=0,n_jobs=8)
    xy_split = get_winedata_split(df,add_rares=True)
    
    dec_tree.fit(np.array(xy_split[0]),np.array(xy_split[2]))
    test_model(dec_tree, np.array(xy_split[1]),np.array(xy_split[3]))
    #xy_split = get_winedata_split(df)
    #rw_model = red_wine_model(xy_split,train=True)
    #test_model(rw_model,np.array(xy_split[1]),np.array(xy_split[3]))
    
    