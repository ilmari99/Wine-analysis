from typing import Tuple, List
import pandas as pd
from scipy.stats import boxcox
import matplotlib.pyplot as plt

def max_norm(df, inplace = True):
    # Normalize data
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

def boxcox_df(df : pd.DataFrame, cols = "all", save_figs=False) -> Tuple[pd.DataFrame,List[Tuple[str,float]]]:
    new_df = df.astype(float)
    # Remove rows with 0 or smaller values
    new_df = new_df.applymap(lambda x : x if x > 0 else pd.NA)
    new_df.dropna(inplace=True,axis=0)
    transformations = []
    if cols == "all":
        cols = df.columns
    for col in cols:
        if save_figs:
            fig,ax = plt.subplots()
            ax.hist(new_df[col])
            ax.set_title(f"Original distribution of {col}")
            plt.savefig(f"original-distribution-{col}")
            plt.close()
        trans = 1
        try:
            new_df[col],trans = boxcox(new_df[col].values,)
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
    plt.cla()
    plt.clf()
    return new_df,transformations