import pandas as pd
import statsmodels.api as sm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import scipy
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from handle import boxcox_df
import linregress
from show import print_model,accuracy_info
from handle import max_norm, make_smogn,add_noise,add_rows_red_wine_regress,add_rows_white_wine_regress,indices_to_one_hot,standard_transformation

def get_winedata_split(df,rm_outliers=False,transform="",norm=False,add_rares = True,onehot=True,wine="red",noise=0,smogn_=False,smoter_kwargs={}):
    # Read and handle data
    boxcox = False
    stand_trans = False
    if "boxcox" in transform:
        boxcox = True
    if "standard" in transform:
        stand_trans = True
    if boxcox and stand_trans:
        raise Warning("Both boxcox and StandardScaler transforms specified")
    endog = df.pop("quality")
    exog = df
    if norm:
        exog = max_norm(exog)
        if not onehot:
            endog = max_norm(pd.DataFrame(endog))
    if stand_trans:
        print("Applying StandardScaler")
        exog = standard_transformation(exog)
    if boxcox:
        df = pd.concat([exog,endog],axis=1)
        cols = list(df.columns)
        cols.remove("quality")
        df,trans = boxcox_df(df,cols=cols)
        endog = df.pop("quality")
        exog = df
    x_train,x_test,y_train,y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    if rm_outliers:
        df = pd.concat([x_train,y_train],axis=1)
        df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
        df = df.reset_index(drop=True)
        y_train = df.pop("quality")
        x_train = df
    if add_rares:
        assert not any([smogn_]), "Cannot add rares if smogn is used. Smogn does the same"
        if wine == "red":
            x_train,y_train = add_rows_red_wine_regress(x_train, y_train,normed=norm)
        elif wine == "white":
            x_train,y_train = add_rows_white_wine_regress(x_train, y_train,normed=norm)
    if noise > 0:
        if noise < 20:
            raise Warning("Noise should be greater than 20! If noise is < 20, then there is a risk of changing the rounded integer. ie applied_noise > 0.5")
        print("Sizes before adding noise: ",y_train.shape, y_test.shape)
        y_train,y_test = add_noise(y_train,y_test,noise_division = noise)
        print("Sizes after adding noise: ",y_train.shape, y_test.shape)
    if smogn_:
        if noise <= 0:
            Warning("Should add noise for smogn algorithm")
        assert not any([add_rares,rm_outliers]), "Not supported combination of arguments"
        x_train, y_train = make_smogn(x_train,y_train,y_header="quality",smoter_kwargs=smoter_kwargs)
    
    if onehot:
        y_train = pd.DataFrame(indices_to_one_hot(y_train,10))
        y_test = pd.DataFrame(indices_to_one_hot(y_test,10))
    return x_train,x_test,y_train,y_test


def test_model(model,x_test,y_test,wine="red"):
    mul = 8 if wine=="red" else 9
    preds = pd.DataFrame(model.predict(x_test))
    y_test = pd.DataFrame(y_test)
    
    fig,ax = plt.subplots()
    errs = mul*(preds-y_test)
    print(errs)
    print(f"Mean absolute error of model: {np.mean(errs,axis=0)[0]}")
    ax.scatter(y_test,errs)
    ax.set_title("Scatter plot of errors: y_obs vs y_pred")
    ax.set_xlabel("Observed value")
    ax.set_ylabel("Predicted value")
    
    fig, ax = plt.subplots()
    ax.hist(errs)
    ax.set_title("Histogram of errors (y_pred - y_obs)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    
    if len(np.shape(np.array(y_test))) > 1 and 1 not in np.shape(np.array(y_test)):
        preds = preds.idxmax(axis=1)
        y_test = y_test.idxmax(axis=1)
    else:
        preds = pd.Series(preds.apply(lambda x : round(mul*x)).to_numpy().flatten())
        y_test = pd.Series(y_test.apply(lambda x : round(mul*x)).to_numpy().flatten())
    #print(preds)
    #print(y_test)
    pred_count = Counter(preds.to_numpy().flatten())
    obs_count = Counter(y_test.to_numpy().flatten())
    conf_mat = confusion_matrix(y_test,preds,labels=list(range(1,11)))
    f1_sc = f1_score(y_test,preds,average="macro")
    print("F1 score of model: ",f1_sc)
    conf_mat = pd.DataFrame(conf_mat,columns=list(range(1,11)),index=list(range(1,11)))
    sns.heatmap(conf_mat,annot=True,fmt=".0f",)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Confusion matrix")
    print("Neural network predictions",pred_count.items())
    print("Actual values",obs_count.items())
    accuracy_info(y_test,preds)
    errs = preds - y_test
    
    fig,ax = plt.subplots()
    ax.bar(list(obs_count.keys()),list(obs_count.values()),label="Observations",alpha=0.75)
    ax.bar(pred_count.keys(),pred_count.values(),label="Predictions",alpha=0.75)
    ax.set_title("Observations and predictions (rounded to the nearest integer) histogram")
    ax.set_xlabel("Observed/predicted value")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    fig,ax = plt.subplots()
    ax.hist(errs)
    ax.set_title("Histogram of prediction errors (round(y_pred) - round(y_obs))")
    ax.set_xlabel("Prediction error (int)")
    ax.set_ylabel("Frequency")
    plt.show()

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
            epochs=80,
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
        tf.keras.layers.Dense(45,activation="linear"),
        tf.keras.layers.Dense(225,activation="relu"),
        tf.keras.layers.Dense(39,activation="tanh"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(1,"sigmoid"),
        ]
    )
    model.build(np.shape(x_train))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0095, amsgrad=True),
        loss=tf.keras.losses.MeanAbsoluteError(),
        )
    if train:
        print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        hist = model.fit(
            x_train, y_train,
            epochs=20,
            verbose=1,
            validation_data=(x_test,y_test),
            batch_size=128,
        )
    else:
        model.load_weights("./koodi/ilmari/white-wine-regress-model.h5")
    return model

def white_wine_forest_regress(exog,endog):
    dec_tree = RandomForestRegressor(n_estimators=256,
                                     random_state=42,
                                     verbose=0,
                                     n_jobs=8,)
    dec_tree.fit(np.array(exog),np.array(endog))
    return dec_tree

def red_wine_forest_regress(exog,endog):
    forest = RandomForestRegressor(n_estimators=100,
                                     random_state=42,
                                     verbose=0,
                                     n_jobs=8,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     
                                     )
    tries = {
        "n_estimators" : [32,50,64,70,90,128,264,300,400,512,1024],
        "max_features" : [1.0,"sqrt"],
        "max_depth" : [2,5,7,9,12,14,16],
        "min_samples_split" : [2,5,10],
        "min_samples_leaf" : [1,2,4],
        "bootstrap" : [True, False],
    }
    #print(sklearn.metrics.get_scorer_names())
    #forest = RandomizedSearchCV(forest,param_distributions=tries,n_iter=100,cv=3,random_state=42,n_jobs=-1,scoring="neg_mean_squared_log_error")
    
    #forest = RandomForestRegressor(n_estimators=256,random_state=42,verbose=0,n_jobs=8,)
    forest.fit(np.array(exog),np.array(endog).flatten())
    print(forest.get_params())
    return forest

def test_whitewine_regress_model(model_type="forest",score="accuracy"):
    # Test on full dataset
    df = pd.read_csv("./viinidata/winequality-white.csv",sep=";")
    
    if model_type == "forest":
        if score == "accuracy":
            # Accuracy 68.8, f1 43
            xy_split = get_winedata_split(df,transform="",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="white",noise=0,smogn_ = False)
        elif score == "f1":
            # accuracy 64, F1 47.4
            smoter_kwargs = {
                "k":256,
                "samp_method":"balance",
                "rel_thres":0.7
            }
            xy_split = get_winedata_split(df,transform="",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="white",noise=80,smogn_ = True,smoter_kwargs=smoter_kwargs)
        model = white_wine_forest_regress(np.array(xy_split[0]),np.array(xy_split[2]))
        
        
        
    elif model_type == "neural":
        if score == "accuracy":
            # Accuracy 56-58, F1 28
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="white",noise=0,smogn_ = False)
        elif score == "f1":
            smoter_kwargs = {
                "k":128,
                "samp_method":"balance",
                "rel_thres":0.7
            }
            # Accuracy 54, F1 32
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="white",noise=80,smogn_ = True,smoter_kwargs=smoter_kwargs)
        model = white_wine_model_regress(xy_split,train=True)
        print_model(model)
        
        
        
    elif model_type == "linear":
        if score == "accuracy":
            # Accuracy 51, f1 23.1
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="white",noise=0,smogn_ = False)
        elif score == "f1":
            # accuracy 44.8, f1 28.4
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=True,onehot=False,wine="white",noise=0,smogn_ = False)
        pops = ["residual sugar","total sulfur dioxide","citric acid","pH","density",]
        #pops = []
        #pops = "default"
        model,pops = linregress.white_wine_linmodel(xy_split[0],xy_split[2], pops = pops)
        xy_split = list(xy_split)
        xy_split[1] = sm.add_constant(xy_split[1]).astype(float)
        [xy_split[1].pop(k) for k in pops]
        
    test_model(model, np.array(xy_split[1]),np.array(xy_split[3]),wine="white")



def test_redwine_regress_model(model_type = "forest",score="accuracy"):
    
    df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
    if model_type == "forest":
        if score == "accuracy":
            # Accuracy 69.7, f1 32.9
            xy_split = get_winedata_split(df,transform="boxcox",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="red",noise=0,smogn_ = False)
        elif score == "f1":
            # Accuracy 64, f1-score 35.8
            smoter_kwargs = {
                "k":256,
                "samp_method":"balance",
                "rel_thres":0.7
            }
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=True,onehot=False,wine="red",noise=0,smogn_ = False,smoter_kwargs=smoter_kwargs)
        model = red_wine_forest_regress(np.array(xy_split[0]),np.array(xy_split[2]))
        
    elif model_type == "neural":
        if score == "accuracy":
            # accuracy 59-63, f1 30-35
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="red",noise=0,smogn_ = False)
        elif score == "f1":
            # accuracy 59-63, f1 30-35
            smoter_kwargs = {
                "k":256,
                "samp_method":"extreme",
                "rel_thres":0.4
            }
            xy_split = get_winedata_split(df,transform="",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="red",noise=80,smogn_ = True,smoter_kwargs=smoter_kwargs)
        model = red_wine_model_regress(xy_split,train=True)
        print_model(model)
        
    elif model_type == "linear":
        if score == "accuracy":
            # Accuracy 62, f1 score 27.5
            xy_split = get_winedata_split(df,transform="",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="red",noise=0,smogn_ = False)
        elif score == "f1":
            # accuracy 54, f1 30
            smoter_kwargs = {
                "k":264,
                "samp_method":"extreme",
                "rel_thres":0.4
            }
            xy_split = get_winedata_split(df,transform="standard",rm_outliers=False,norm=True,add_rares=False,onehot=False,wine="red",noise=80,smogn_ = True,smoter_kwargs=smoter_kwargs)
        pops = ["residual sugar","citric acid", "fixed acidity", "density","free sulfur dioxide"]
        print(xy_split[0],xy_split[2])
        model,pops = linregress.red_wine_linmodel(xy_split[0],xy_split[2], pops = pops)
        xy_split = list(xy_split)
        xy_split[1] = sm.add_constant(xy_split[1]).astype(float)
        [xy_split[1].pop(k) for k in pops]
        
    test_model(model, np.array(xy_split[1]),np.array(xy_split[3]),wine="red")

    
if __name__ == "__main__":
    test_redwine_regress_model(model_type="linear",score="accuracy")
    
    