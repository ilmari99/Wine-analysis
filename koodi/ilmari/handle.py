from typing import Tuple, List
import pandas as pd
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import smogn
import random
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

def make_smogn(train : pd.DataFrame ,y_train : pd.Series = None,y_header = "",smoter_kwargs = {}):
    # If there is no y data specified
    train = train.reset_index(drop=True)
    print("Train: \n",train)
    if y_train is None:
        if not y_header:
            raise ValueError("Specify the y data by y_header or y_train data")
    # If there is y data specified
    else:
        y_train = y_train.reset_index(drop=True)
        print("y data: \n", y_train)
        # If the y data is not a series
        if not isinstance(y_train,pd.Series):
            # If the y data is not a series and no y_header given
            if not y_header:
                raise ValueError("y_header can only be inferred from the y_train data if y_train is a Series")
            if isinstance(y_train,pd.DataFrame):
                assert len(y_train.columns.values) == 1, "If the input is a dataframe, it mut only contain 1  column"
                y_train = y_train.squeeze()
            y_train = pd.Series(y_train,name=y_header)
        
        y_header = y_train.name
        train = pd.concat([train,y_train],axis=1)
    #x_train = x_train.reindex()
    print("trainset:\n",train)
    print(train.columns)
    print("'quality' loc: ",train.columns.get_loc(y_header))
    smoter_kwargs.setdefault("k",256)
    smoter_kwargs.setdefault("samp_method","balance")
    smoter_kwargs.setdefault("rel_thres",0.7)
    print(smoter_kwargs)
    train = smogn.smoter(data=train, y=y_header,**smoter_kwargs,)
    return train, train.pop(y_header)

def add_noise(y_train,y_test, noise_division = 20):
    def func(x):
        pm = 1 if random.random() > 0.5 and x < 1 else -1
        noise = random.random() / noise_division
        return x + pm*noise
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    y_train = y_train.applymap(func)
    y_test = y_test.applymap(func)
    return y_train, y_test


def max_norm(df, inplace = True):
    # Normalize data
    df = df.astype(float)
    if not inplace:
        df = df.copy()
    df_max_min = [(max(df[col]),min(df[col])) for col in df.columns]
    df = df.apply(lambda x : x / max(x))
    return df

def concat_clean_separate(*args):
    """Concatenates the given dataframes, drops NaN values, and returns the dataframes separated to a tuple
    """
    #args = [pd.DataFrame(data=a,columns=[a.columns if isinstance(a,pd.DataFrame) else a.Name]) for a in args]
    args = [pd.DataFrame(a) for a in args]
    try:
        headers = [df.columns for df in args]
    except AttributeError:
        raise AttributeError("All arguments must be dataframes")
    df = pd.concat(args,axis=1)
    df.dropna(inplace=True)
    dfs = [df.loc[:,h] for h in headers]
    return tuple(dfs)


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

def standard_transformation(df):
    scaler = StandardScaler()
    cols = list(df.columns)
    scaler.fit_transform(df)
    df = pd.DataFrame(scaler.transform(df),columns=cols)
    return df

def boxcox_df(df : pd.DataFrame, cols = "all", save_figs=False) -> Tuple[pd.DataFrame,List[Tuple[str,float]]]:
    new_df = df.astype(float)
    # Remove rows with 0 or smaller values
    new_df = new_df.applymap(lambda x : x if x > 0 else pd.NA)
    new_df.dropna(inplace=True,axis=0)
    transformations = []
    if cols == "all":
        cols = df.columns
    new_df = new_df.astype(float)
    for col in cols:
        if save_figs:
            fig,ax = plt.subplots()
            ax.hist(new_df[col])
            ax.set_title(f"Original distribution of {col}")
            plt.savefig(f"original-distribution-{col}")
            plt.close()
        trans = 1
        try:
            new_df[col],trans = boxcox(new_df[col].values)
        except TypeError as te:
            print(f"column {col}: {te}")
        print(f"transformed distribution of {col} with lambda: {trans}")
        if save_figs:
            fig,ax = plt.subplots()
            ax.hist(new_df[col])
            ax.set_title(f"Transformed distribution of {col} with lambda: {trans}")
            plt.savefig(f"transformed-distribution-{col}")
            plt.close()
        transformations.append((col,trans))
    new_df = new_df.astype(float)
    plt.cla()
    plt.clf()
    return new_df,transformations