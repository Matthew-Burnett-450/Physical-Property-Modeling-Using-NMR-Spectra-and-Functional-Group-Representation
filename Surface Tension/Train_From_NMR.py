import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from Util import load_data, fit_fourier_series,cutoff_point_detector,fit_exp_curve
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV,LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.svm import SVR
import gtda.diagrams as diag
from scipy.stats import skew, kurtosis
import sweetviz as sv
from sklearn.preprocessing import StandardScaler
# Load data
X_point_cloud, FG, T, Y, FG_labels,idx,Names,Tc,Tc_True = load_data()
#for each T divide by Tc
for i in range(len(T)):
    tc=Tc[i]
    for j in range(len(T[i])):
        T[i][j] = T[i][j]/tc

#if X_point_cloud is negative or empty, remove from T and Y
X_point_cloud_new = []
T_new = []
Y_new = []
idx_new = []

for i in range(len(X_point_cloud)):
    if len(X_point_cloud[i])>0:
        X_point_cloud_new.append(X_point_cloud[i])
        T_new.append(T[i])
        Y_new.append(Y[i])
        idx_new.append(idx[i])
X_point_cloud = X_point_cloud_new
T = T_new
Y = Y_new
idx = idx_new


#debug
"""for y,t,id in zip(Y,T,idx):
    print(id[0])
    plt.plot(t,y)
    plt.show()
plt.close()"""

#bin point cloud on histogram 
X_point_cloud_new = []  
for i in range(len(X_point_cloud)):
    x = np.array(X_point_cloud[i])
    x = x[:,0]
    hist, bin_edges = np.histogram(x, bins=250,range=(0,250),density=True)
    temp_vec = []
    for i in range(len(hist)):
        x_coord = i
        y_coord = hist[i]
        temp_vec.append([x_coord,y_coord,1])
    X_point_cloud_new.append(temp_vec)
X_point_cloud = np.array(X_point_cloud_new)



#grab stats from X_point_cloud , mean max min std, skew, kurtosis
X_stats = []
for i in range(len(X_point_cloud)):
    temp_vec = []
    
    temp_vec.append(np.mean(X_point_cloud[i][:,1][np.nonzero(X_point_cloud[i][:,1])]))
    temp_vec.append(np.max(X_point_cloud[i][:,1][np.nonzero(X_point_cloud[i][:,1])]))
    temp_vec.append(np.min(X_point_cloud[i][:,1][np.nonzero(X_point_cloud[i][:,1])]))
    temp_vec.append(np.std(X_point_cloud[i][:,1][np.nonzero(X_point_cloud[i][:,1])]))
    temp_vec.append(np.average(X_point_cloud[i][:,0],weights=X_point_cloud[i][:,1]))
    temp_vec.append(skew(X_point_cloud[i])[1])
    temp_vec.append(np.max(X_point_cloud[i][:,1][np.nonzero(X_point_cloud[i][:,1])])-np.min(X_point_cloud[i][:,1][np.nonzero(X_point_cloud[i][:,1])]))

    X_stats.append(temp_vec)
X_stats = np.array(X_stats)

if False:
    X_point_cloud_new = []
    for j in range(len(X_point_cloud)):
        #convert to numpy array 3D
        i = X_point_cloud[j]
        i = np.expand_dims(i, axis=0)
        print(i.shape)
        X_point_cloud_new.append(i)
    X_point_cloud = np.array(X_point_cloud_new)


    #grab ampltiude diagrams from diag
    metrics = ['bottleneck','persistence_image']
    Amp_Vec = []
    for i in range(len(X_point_cloud)):
        temp_vec = []
        for metric in metrics:
            diagrams = diag.Amplitude(metric=metric, n_jobs=8).fit_transform(X_point_cloud[i])
            temp_vec.append(diagrams[0][0])
        Amp_Vec.append(temp_vec)
        print(i/len(X_point_cloud)*100)

    #duplicate X for each idx
    X=[]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            X.append(Amp_Vec[i])   
    X = np.array(X)
    np.save('X.npy',X)

#if negative, remove from T and Y
X=np.load('X.npy')


#duplicate X_stats for each idx
X_stats_new=[]
for i in range(len(Y)):
    for j in range(len(Y[i])):
        X_stats_new.append(X_stats[i])
X_stats = np.array(X_stats_new)

#add stats to X
X = np.hstack((X,X_stats))

#flatten Y
T = [item for sublist in T for item in sublist]
Y = [item for sublist in Y for item in sublist]
idx = [item for sublist in idx for item in sublist]


#save X
X = np.array(X)
Y = np.array(Y)*1000 
Y=Y.ravel()
T = np.array(T)
idx = np.array(idx)


T = T.reshape(-1,1)
X = np.hstack((X,T))

print(X.shape)
print(Y.shape)

#LOOCV
logo = LeaveOneGroupOut()
mape = []
MAE_ls=[]
MAPE_ls=[]
LOO_Result = []
Y_test_ls = []
Y_pred_ls = []

for train_inds, test_inds in logo.split(X, Y, idx):
    X_train, X_test = X[train_inds], X[test_inds]
    Y_train, Y_test = Y[train_inds], Y[test_inds]
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    mape=np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
    MAE=np.mean(np.abs((Y_test - y_pred)))
    R2 = model.score(X_test, Y_test)
    LOO_Result.append([Names[idx[test_inds[0]]],MAE,mape])
    MAE_ls.append(MAE)
    MAPE_ls.append(mape)
    Y_test_ls.append(Y_test)
    Y_pred_ls.append(y_pred)

#calculate the mean of MAE and MAPE
MAPE_mean = np.mean(MAPE_ls)
Max_MAPE = np.max(MAPE_ls)
Min_MAPE = np.min(MAPE_ls)
print(f'LOOCV Mean Absolute Percentage Error: {MAPE_mean:.2f}%')
print(f'LOOCV Max Absolute Percentage Error: {Max_MAPE:.2f}%')
print(f'LOOCV Min Absolute Percentage Error: {Min_MAPE:.2f}%')

#save LOO_Result to xlsx
df = pd.DataFrame(LOO_Result, columns=['Name','MAE','MAPE'])
df.to_excel('LOO_Result_NMR.xlsx',index=False)


#plot the Y
plt.figure(figsize=(6,5))
for y_test,y_pred,Name in zip(Y_test_ls,Y_pred_ls,Names):
    plt.scatter(y_test,y_pred,s=3,label=f'{Name}')
plt.legend()
plt.xlabel('True Surface Tension')
plt.ylabel('Predicted Surface Tension')
plt.xlim([0, 50])
plt.ylim([0, 50])
plt.plot([0, 50], [0, 50], ls='--', c='k')
plt.title('True vs Predicted Surface Tension LOOCV')
plt.savefig('Figs\True_vs_Predicted_Surface_Tension_LOOCV_NMR.png')
plt.show()
