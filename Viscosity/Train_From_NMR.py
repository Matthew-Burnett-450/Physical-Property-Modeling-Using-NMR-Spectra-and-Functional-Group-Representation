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
X_point_cloud, FG, T, Y, FG_labels,idx,Names = load_data()
Names=np.array(Names)

print(len(Names))
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

if True:
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
        print((i/len(X_point_cloud))*100)

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
#X = X_stats



#flatten Y
T = [item for sublist in T for item in sublist]
Y = [item for sublist in Y for item in sublist]
idx = [item for sublist in idx for item in sublist]


#save X
X = np.array(X)
Y = np.array(Y) * 1000
T = np.array(T)
idx = np.array(idx)



T = T.reshape(-1,1)
X = np.hstack((X,T))

Y=np.log(Y)



print(X.shape)
print(Y.shape)

# Split data into training and testing sets
train_inds, test_inds = next(GroupShuffleSplit(test_size=0.1, n_splits=2).split(X, Y, idx))

X_train, X_test = X[train_inds], X[test_inds]
Y_train, Y_test = Y[train_inds], Y[test_inds]

# Train a linear regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Evaluate the model
train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print(f'Training R^2: {train_score:.2f}')
print(f'Testing R^2: {test_score:.2f}')

#calculate MAPE
y_pred = model.predict(X_test)
mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
print(f'Mean Absolute Percentage Error: {mape:.2f}%')


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
    Y_test = np.exp(Y_test)
    y_pred = np.exp(y_pred)
    mape=np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
    MAE=np.mean(np.abs((Y_test - y_pred)))
    print(f'LOOCV MAE: {MAE:.2f},Name: {Names[idx[test_inds[0]]]}, MAPE: {mape:.2f}%')
    LOO_Result.append([Names[idx[test_inds[0]]],MAE,mape])
    MAE_ls.append(MAE)
    MAPE_ls.append(mape)
    Y_test_ls.append(Y_test)
    Y_pred_ls.append(y_pred)
print(f'Mean MAE: {np.mean(MAE_ls):.2f}, Mean MAPE: {np.mean(MAPE_ls):.2f}%')

#save LOO_Result to xlsx
df = pd.DataFrame(LOO_Result, columns=['Name','MAE','MAPE'])
df.to_excel('LOO_Result_NMR.xlsx',index=False)

#add the MAE to the json file
with open('ViscTrainingData.json') as f:
    data = json.load(f)
for i in range(len(data)):
    data[i]['MAE'] = MAE_ls[i]
    data[i]['MAPE'] = MAPE_ls[i]
with open('ViscTrainingData.json', 'w') as f:
    json.dump(data, f)

#plot the Y
plt.figure()
for y_test,y_pred in zip(Y_test_ls,Y_pred_ls):
    plt.scatter(y_test,y_pred,s=3)
plt.xlabel('True Viscosity')
plt.ylabel('Predicted Viscosity')
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.plot([0, 20], [0, 20], ls='--', c='k')
plt.title('True vs Predicted Viscosity LOOCV')
plt.show()



#load jetfuel
X_jetfuel = np.load('X_jetfuels.npy')
Y_jetfuel = np.load('Y_jetfuels.npy')
T_jetfuel = np.load('T_jetfuels.npy')
idx_jetfuel = np.load('idx_jetfuels.npy')

X_jetfuel = np.hstack((X_jetfuel,T_jetfuel.reshape(-1,1)))

#make predictions
Y_pred_jetfuel = model.predict(X_jetfuel)

Y_pred_jetfuel = np.exp(Y_pred_jetfuel)

#fit a curve of the form exp(A + B/T ) to each 3
B_1,A_1 = fit_exp_curve(T_jetfuel[0:2],Y_jetfuel[0:2])
B_2,A_2 = fit_exp_curve(T_jetfuel[:-3],Y_jetfuel[:-3])

Y_pred_jetfuel[0] = np.exp(A_1 + B_1/T_jetfuel[0])
Y_pred_jetfuel[1] = np.exp(A_1 + B_1/T_jetfuel[1])
Y_pred_jetfuel[2] = np.exp(A_1 + B_1/T_jetfuel[2])

Y_pred_jetfuel[-1] = np.exp(A_2 + B_2/T_jetfuel[-1])
Y_pred_jetfuel[-2] = np.exp(A_2 + B_2/T_jetfuel[-2])
Y_pred_jetfuel[-3] = np.exp(A_2 + B_2/T_jetfuel[-3])

#MAPE
MAPE = np.median(np.abs((Y_jetfuel - Y_pred_jetfuel) / Y_jetfuel)) * 100
print(f'MAPE: {MAPE:.2f}%')
#r2
from sklearn.metrics import r2_score
r2 = r2_score(Y_jetfuel,Y_pred_jetfuel)
print(f'R2: {r2:.2f}')

#parity plot
plt.scatter(Y_jetfuel,Y_pred_jetfuel,c='blue',s=5)
plt.plot([0, 40], [0, 40],ls="--", c='k')
plt.xlim([0,40])
plt.ylim([0,40])
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Parity Plot')
plt.show()

#plot temp vs viscosity
#plot first 7 and last 7
Names=['HRJ POSF 7720','Jet-A POSF 10325']
gen=0
for i in [[0,1,2],[-1,-2,-3]]:
    
    plt.scatter(T_jetfuel[i],Y_jetfuel[i],label=f'True {Names[gen]}')
    plt.plot(T_jetfuel[i],Y_pred_jetfuel[i],label='Predicted',ls='--')

    gen+=1

plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (mPa.s)')
plt.legend()
plt.title('Prediction of Real Fuels')
plt.show()









