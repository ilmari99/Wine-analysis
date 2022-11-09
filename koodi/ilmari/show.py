import matplotlib.pyplot as plt

def plot_models(df,y_header="quality"):
    # Create graphs for the normalized original variables
    ms = ModelSet(df,y_header=y_header)
    for col in df.columns:
        if col != y_header:
            plot_model(ms.models[col], df[col], df[y_header],has_constant=True,show=False)
        plt.savefig(f"original-{col}-{y_header}")
        plt.close()
    return

def plot_model(model,x,y, show=True,save=None,has_constant=False):
    """ Plot a model alongside the data.
    Copies the x and y variables
    Checks if x variable has a column named 'const'
        If yes, drops the 'const' columns for easy plotting.
    Plots the x and y variables in a scatter plot.
    If the x variable had a 'const' column, or if has_constant is True,
        add a column of ones to the x variable.
    predict the y variable using the model and plot the predicted values.
    If save is not None, saves the plot to a file specified by the save variable.
    """
    assert len(x) == len(y)
    x,y = x.copy(),y.copy()
    if isinstance(x,pd.DataFrame) and "const" in x.columns:
        has_constant = True
        x.drop("const",axis=1,inplace=True)
    xname = x.columns[0] if isinstance(x,pd.DataFrame) else x.name
    yname = y.columns[0] if isinstance(y,pd.DataFrame) else y.name
    plt.figure()
    plt.scatter(x,y,label="data")
    plt.title(f"{xname} vs {yname}")
    plt.xlabel(xname)
    plt.ylabel(yname)
    if has_constant:
        x = sm.add_constant(x)
    ypred = model.predict(x)
    plt.plot(x.pop(xname),ypred,"r",label="prediction")
    plt.legend()
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()