import json
import numpy as np
from sklearn.metrics import r2_score
def get_data():
    with open('DataCompolation\FGTrainingDB.json') as f:
        data = json.load(f)
    
    temprature = []
    y_ls = []
    FG = []
    A = []
    B = []
    Names = []
    for item in data:
        temprature.append(item['t'])
        y_ls.append(item['y'])
        FG.append(item['FG'])
        y = item['y']
        y = np.array(y)
        y = np.log(y)
        t = item['t']
        t = np.array(t)

        y = y[t > 250]
        t = t[t > 250]
        y = y[t < 600]
        t = t[t < 600]
        
        # Fit A + B*t
        a = np.ones((len(t), 1))
        b = t.reshape(-1, 1)
        X = np.concatenate((a, 1 / b), axis=1)
        linear = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X.dot(linear[0])
        r2 = r2_score(y, y_hat)
        A.append(linear[0][0])
        B.append(linear[0][1])
        Names.append(item['name'])

    


    t = np.array(temprature)
    y = np.array(y_ls)
    FG = np.array(FG)
    A = np.array(A)
    B = np.array(B)
    Names = np.array(Names)
    Target = np.concatenate((A.reshape(-1, 1), B.reshape(-1, 1)), axis=1)

    return FG, Target,t,y,Names

