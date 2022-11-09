from handle import boxcox_df,max_norm
from show import plot_model
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy

if __name__ == "__main__":
    df = pd.read_csv("./viinidata/winequality-red.csv",sep=";")
    #df["quality"] = df["quality"].apply(lambda x : pd.NA if ((int(x) in [5,6]) and (random.random() > 0.5)) else x)
    #df["quality"] = df["quality"].apply(lambda x : pd.NA if ((int(x) in [5,6]) and (random.random() > 0.1)) else x)
    