import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

def pearson_loss(y_true, y_pred):


    y_bar = y_true.mean()
    yhat_bar = y_pred.mean()
    x_xm = (y_true - y_bar)
    w_wm = (y_pred - yhat_bar)+0.00001
    w_wm2 = ((y_pred - yhat_bar) ** 2)

    B = ((y_true - y_bar) * (y_pred - yhat_bar))
    C = np.sqrt(((y_true - y_bar) ** 2))  # constant variable
    D = np.sqrt(((y_pred - yhat_bar) ** 2))+0.00001 # std of pred
    gradient = (x_xm - ((B / D) * w_wm)) / np.sqrt(C * D)
    hessian = (3 * B * w_wm2 / (D ** 2) - (x_xm * w_wm + B) / D - (x_xm * w_wm)) / np.sqrt(C * D)
    print('\n\n',x_xm, '\n', w_wm,'\n',D ,'\n',y_true,'\n',y_pred,'\n\n',gradient)
    return gradient, hessian


def metric(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    print(y_pred)
    n = len(y_true)
    y_bar = y_true.mean()
    yhat_bar = y_pred.mean()
    B = ((y_true - y_bar) * (y_pred - yhat_bar)).sum()
    C = np.sqrt(((y_true - y_bar) ** 2)).sum()  # constant variable
    D = np.sqrt(((y_pred - yhat_bar) ** 2)).sum()
    metric=B/np.sqrt(C*D)
    return 'pearson corr',metric

filename = ['allnans_dropped_train.csv', 'allnans_dropped_test.csv']
df = pd.read_csv(filename[0])
targe_col_name = 'SEVERITYCODE'
x = df.drop(targe_col_name, axis=1)
y = df[targe_col_name]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
y_test-=1
y_train-=1
dtrain = xgb.DMatrix(X_train, y_train)


def mse_approx_obj(dtrain, preds):
    d = preds - dtrain
    grad_mse = d
    hess_mse = np.full(d.shape,1.0)

    return grad_mse, hess_mse

def mae_approx_obj(dtrain, preds):
    d = preds - dtrain
    grad_mae = np.array(d)
    grad_mae[grad_mae > 0] = 1.
    grad_mae[grad_mae <= 0] = -1.
    hess_mae = np.full(d.shape, 0.0)

    return grad_mae, hess_mae

def pseudohuber_approx_obj(dtrain,preds):

    d = preds- dtrain
    h = 1  #h is the delta
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt

    return grad, hess
def pearson_corr(predt: np.ndarray, dtrain: xgb.DMatrix):
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    y_bar = np.mean(y)
    yhat_bar = predt.mean()
    B = ((y - y_bar) * (predt - yhat_bar)).sum()
    C = np.sqrt(((y - y_bar) ** 2)).sum()  # constant variable
    D = np.sqrt(((y - yhat_bar) ** 2)).sum()
    metric = B / np.sqrt(C * D)
    return 'PearsonCorr', metric

def model(x_train,y_train,x_test, y_test):

     xgb1 = xgb.XGBRegressor(n_jobs=-1,n_estimators =200,learning_rate=0.001, scale_pos_weight=2.3,objective = pseudohuber_approx_obj, seed=2)

     xgb1.fit(x_train,y_train, eval_set=[(x_train,y_train), (x_test,y_test)], eval_metric=pearson_corr, verbose=True)

     return xgb1

xgb1 = model(X_train, y_train,X_test, y_test)