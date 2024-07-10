import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Util import *
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.preprocessing import PolynomialFeatures 
import sweetviz as sv
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
import json
from scipy.optimize import minimize
_, FG_desciprtors, T, SurfT, FG_labels,ids,Names,Tc,Tc_True = load_data("SurfTTrainingData_full_iso.json")
Tc=Tc_True
X=[]
Y=[]
Temp=[]
idx=[]
T=np.array(T)

for i in range(len(T)):
    tc=Tc[i]
    for t,y in zip(T[i],SurfT[i]):
        X.append(FG_desciprtors[i])
        Temp.append(t/tc)
        Y.append(y)
    plt.plot(T[i]/tc,SurfT[i],label=Names[i])
plt.xlabel('Tc / Temperature (K)')
plt.ylabel('Surface Tension (mN/m)')
plt.legend()
plt.show()

#flatten ids
idx=[item for sublist in ids for item in sublist]
idx=np.array(idx)
X = np.array(X)
#remove all zero columns

#sweetviz
if False:
    X_df = pd.DataFrame(X, columns=FG_labels)
    Y_df = pd.DataFrame(Y, columns=['Surface Tension'])
    X_df = pd.concat([X_df, Y_df], axis=1)
    report = sv.DataframeReport(X_df,target_feature_name='Surface Tension')
    report.show_html()

T=np.array(Temp).reshape(-1,1)
X=np.hstack((X,T*X))
poly=PolynomialFeatures(degree=1,include_bias=False)
X=poly.fit_transform(X)
Y = np.array(Y)*1000


#fit 
model = LinearRegression(fit_intercept=True)
model.fit(X, Y)


#predict
y_pred = model.predict(X)
#calculate MAE
mae = np.mean(np.abs((Y - y_pred)))
mape = np.mean(np.abs((Y - y_pred) / Y)) * 100
r2 = 1 - np.sum((Y - y_pred) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
print(f'Mean Absolute Percentage Error: {mape:.2f}%')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R2: {r2:.2f}')

for Name,id in zip(Names,ids):
    id = np.where(idx==id[0])[0]
    plt.scatter(y_pred[id],Y[id],label=Name)
plt.plot([0,50],[0,50],ls='--',c='k')
plt.xlabel('True Surface Tension')
plt.ylabel('Predicted Surface Tension')
plt.legend(fontsize=6)
plt.show()


#LOOCV
logo = LeaveOneGroupOut()
LOO_Result = []
MAE_ls = []
MAPE_ls = []
Y_test_ls = []
Y_pred_ls = []
Temperature_list = []
i = 0
for train_inds, test_inds in logo.split(X, Y, idx):
    X_train, X_test = X[train_inds], X[test_inds]
    Y_train, Y_test = Y[train_inds], Y[test_inds]
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    #print the coefficients
    #print(model.intercept_,model.coef_)
    y_pred = model.predict(X_test)
    
    mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
    mae = np.mean(np.abs((Y_test - y_pred)))
    r2 = 1 - np.sum((Y_test - y_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2)
    
    LOO_Result.append([
        Names[idx[test_inds[0]]], mae, mape,
        FG_desciprtors[i][0], FG_desciprtors[i][1], FG_desciprtors[i][2],
        FG_desciprtors[i][3], FG_desciprtors[i][4], FG_desciprtors[i][5],
        FG_desciprtors[i][6]
    ])
    
    MAE_ls.append(mae)
    MAPE_ls.append(mape)    
    Y_test_ls.append(Y_test)
    Y_pred_ls.append(y_pred)
    Temperature_list.append(T[test_inds])
    i += 1

#save each temprature and Y curve to xlsx
df = pd.DataFrame()
for i in range(len(Y_test_ls)):
    df['Y_true'+Names[i]] = Y_test_ls[i]
    df['Y_pred'+Names[i]] = Y_pred_ls[i]
    df['T'+Names[i]] = Temperature_list[i]
df.to_excel('LOOCV_FG_Temp.xlsx',index=False)



#calculate the mean of MAE and MAPE
MAPE_mean = np.mean(MAPE_ls)
Max_MAPE = np.max(MAPE_ls)
Min_MAPE = np.min(MAPE_ls)

print(f'LOOCV Mean Absolute Percentage Error: {MAPE_mean:.2f}%')
print(f'LOOCV Max Absolute Percentage Error: {Max_MAPE:.2f}%')
print(f'LOOCV Min Absolute Percentage Error: {Min_MAPE:.2f}%')
print(f'LOOCV Mean Absolute Error: {np.mean(MAE_ls):.2f}')
#save LOO_Result to xlsx
df = pd.DataFrame(LOO_Result, columns=['Name','MAE','MAPE','CH3','CH','Quaternary_C','Benzyl_Rings','CH2','CH2_Chains','Cyclo_CH2'])
df.to_excel('LOO_Result_FG.xlsx',index=False)

#add the MAE to the json file
with open('SurfTTrainingData_full.json') as f:
    data = json.load(f)
for i in range(len(data)):
    data[i]['MAE'] = MAE_ls[i]
    data[i]['MAPE'] = MAPE_ls[i]
with open('SurfTTrainingData_full.json', 'w') as f:
    json.dump(data, f)

#plot the Y
plt.figure(figsize=(6,5))
for y_test,y_pred,Name in zip(Y_test_ls,Y_pred_ls,Names):
    plt.scatter(y_test,y_pred,s=5,label=Name)
plt.scatter([0],[0],label=f'Max MAPE {Max_MAPE:.2f}%',marker=None,visible=False,s=5)
plt.scatter([0],[0],label=f'Min MAPE {Min_MAPE:.2f}%',marker=None,visible=False,s=5)
plt.scatter([0],[0],label=f'Mean MAPE {MAPE_mean:.2f}%',marker=None,visible=False,s=5)
plt.xlabel('True Surface Tension')
plt.ylabel('Predicted Surface Tension')
plt.xlim([0, 50])
plt.ylim([0, 50])
plt.plot([0, 50], [0, 50], ls='--', c='k')
#two column legend
plt.legend(ncol=2,fontsize=4,loc='upper left')
#add text with Min, Max and Mean MAPE to legend

plt.title('True vs Predicted Surface Tension LOOCV')
plt.savefig('Figs\True_vs_Predicted_Surface_Tension_LOOCV_FG_IsoAlkanes.png')
plt.show()

