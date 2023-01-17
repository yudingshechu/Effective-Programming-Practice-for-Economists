import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def calculate_diagnostics(model, x_test, y_test):
    """model validation.

    Args:
        model (_type_): _description_
        x_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_

    """
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1].flatten()
    out = {
        "mse": mean_squared_error(y_test, probs),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
    }
    return out


def model_building_up(data):
    formula = "smoke~gender+marital_status+highest_qualification+gross_income+ethnicity+nationality+region+age"
    formula = "+".join([formula] + [f"I(age**{i})" for i in range(2, 11)])
    y, x = dmatrices(formula, data=data, return_type="dataframe")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1899
    )
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x.columns)
    penalties=list(np.linspace(0.001,5,num=100))
    models={
        p:LogisticRegression(fit_intercept=False,penalty="l2",C=1/p,max_iter=2000).fit(x_train_scaled,y_train)
    for p in penalties
    }
    return models, x_test_scaled, y_test