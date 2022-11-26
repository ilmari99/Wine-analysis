from collections import Counter
import matplotlib.pyplot as plt
from ModelSet import ModelSet
import statsmodels.api as sm
import pandas as pd
import tensorflow as tf

_LAYER_KEYS = {
    "dense":["units","activation"],
    "dropout":["rate"],
    "conv2d":["filters","kernel_size","activation","strides"],
    "max_pooling2d":["pool_size","strides"],
    "flatten":["dtype"],
}

def accuracy_info(y_test, preds):
    obs_count = Counter(y_test.to_numpy().flatten())
    hits = {}#0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
    for obs, pred in zip(y_test.values,preds.values):
        if obs == pred:
            obs = int(obs)
            if obs not in hits:
                hits[obs] = 0
            hits[obs] += 1
    corrects = 0
    for qual,count in hits.items():
        corrects += count
        print(f"Accuracy of model for quality {qual}: ", round(count/obs_count[qual],3))
    print(f"Total accuracy of model: {corrects/len(y_test)}")

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
        

def create_dict(model: tf.keras.models.Sequential,learning_metrics={}) -> dict:
    dic = {
        "optimizer":None,
        "loss_function":None,
        "layers":{},
        "learning_metrics":learning_metrics,
    }
    if model.optimizer != None:
        dic["optimizer"] = model.optimizer.get_config()
    else:
        raise AttributeError("Model must be compiled before creating a dictionary.")
    if isinstance(model.loss,str):
        dic["loss_function"] = model.loss
    else:
        dic["loss_function"] = model.loss.__class__.__name__
    layer_configs = [layer.get_config() for layer in model.layers]
    layers_summary = {}
    for i,config in enumerate(layer_configs):
        name = config.get("name")
        layers_summary[name] = {}
        keys = []
        for layer_name in _LAYER_KEYS.keys():
            if layer_name in name:
                keys = _LAYER_KEYS[layer_name]
                break
        if not keys:
            print(f"Layer {name} has not been implemented yet.")
        for key in keys:
            layers_summary[name][key] = config.get(key)
    #Now: layers_summary = {
        # "ccconv2d":{"filter":16,"kernel_size":(2,2),"activation":"relu"}
        # "dense_1":{"units":1,"activation":"relu"},
        # }
    for layer in layers_summary:
        dic["layers"][layer] = layers_summary[layer]
    return dic

def _string_format_model_dict(dic: dict):
    string = f"\nOptimizer: {dic.get('optimizer')}\n"
    string = string + f"Loss function: {dic.get('loss_function')}\n"
    string = string + f"Learning metrics: {dic.get('learning_metrics')}\n"
    for layer in dic["layers"]: #Here variable layer is the name of the layer
        string = string + f"{layer:<16}"
        keys = list(dic["layers"][layer].keys())
        for key in keys:
            string = string + f"{key:<16}:{str(dic['layers'][layer][key]):<16}"
        string = string + "\n"
    return string

def print_model(dic, learning_metrics={}):
    """Takes in a a dict or a model. If argument is model, creates a dictionary from that model that is then used to print the model.

    Args:
        dic (dict or model): dictionary or model to be printed to stdout
        learning_metrics (dict, optional): Dictionary with learning metrics for example {"LAST_LOSS":0.03567}. Defaults to {}.

    Raises:
        TypeError: dic is not a Sequential model or a dictionary
    """    
    if isinstance(dic,tf.keras.models.Sequential):
        dic = create_dict(dic,learning_metrics=learning_metrics)
    if not isinstance(dic, dict):
        raise TypeError(f"Excpected argument of type Sequential or dict, but received {type(dic)}")
    string = _string_format_model_dict(dic)
    print(string)