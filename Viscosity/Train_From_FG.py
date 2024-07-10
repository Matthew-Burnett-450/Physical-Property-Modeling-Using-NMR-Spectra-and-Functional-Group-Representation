import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Util import load_data
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
import sweetviz as sv
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
import json

_, FG_desciprtors, T, SurfT, FG_labels,ids,Names = load_data()


X=[]
Y=[]
idx=[]
for i in range(len(T)):
    for t,y in zip(T[i],SurfT[i]):
        X.append([t]+FG_desciprtors[i])
        Y.append(y)
#flatten ids
idx=[item for sublist in ids for item in sublist]
idx=np.array(idx)
X = np.array(X)
Y = np.log(np.array(Y)*1000)
print(np.max(idx))



print(X.shape)
print(Y.shape)
print(idx.shape)

#sweetviz
if False:
    X_df = pd.DataFrame(X, columns=['T']+FG_labels)
    Y_df = pd.DataFrame(Y, columns=['Viscosity'])
    X_df = pd.concat([X_df, Y_df], axis=1)
    report = sv.DataframeReport(X_df)
    report.show_html()

# Split data into training and testing sets
train_inds, test_inds = next(GroupShuffleSplit(test_size=0.1, n_splits=2).split(X, Y, idx))

X_train, X_test = X[train_inds], X[test_inds]
Y_train, Y_test = Y[train_inds], Y[test_inds]

# Train a linear regression model
model = RandomForestRegressor(n_estimators=5000,random_state=42)
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

Y_test = np.exp(Y_test)
y_pred = np.exp(y_pred)

#plot

#plot by idx
idx_tests = idx[test_inds]
idx_tests = np.unique(idx_tests)
for i in idx_tests:
    idx_test = np.where(idx==i)
    #find the index of the idx_tests in the test_inds fror all idx
    idx_test = np.intersect1d(idx_test, test_inds,return_indices=True)[2]
    plt.scatter(Y_test[idx_test], y_pred[idx_test], label=Names[i])
plt.xlabel('True Viscosity')
plt.ylabel('Predicted Viscosity')
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.plot([0, 20], [0, 20], ls='--', c='k')
plt.legend()
plt.title('True vs Predicted Viscosity')
plt.show()


"""#LOOCV
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
df.to_excel('LOO_Result.xlsx',index=False)

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
plt.show()"""

#predict real fuels
fuels = {
    'Jet-A POSF 10325': [[3.335467977, 1.140190557, 1.055652435, 1.590017098, 3.188735797, 0, 0],[273.15-40,273.15-20,40+273.15],[9.55,4.70,1.80]],
    'HRJ POSF 7720': [[3.271797853, 1.245511112, 1.393100123, 2.148655347, 3.720826794, 0, 0],[233.15,253.15,313.15],[14.00,6.10,1.50]],
}
# Trained in ['CH3','CH','Quaternary_C','Benzyl_Rings','CH2','CH2_Chains','Cyclo_CH2']
# fuels list is [CH2n,CH,CH2 cyclo,CH2,CH3,Quaternary_C,Benzyl_Rings]

#constuct X
X_jetA = [[273.15-40,3.335467977, 1.140190557, 1.055652435, 1.590017098, 3.188735797, 0, 0.192140809],[273.15-20,3.335467977, 1.140190557, 1.055652435, 1.590017098, 3.188735797, 0, 0],[40+273.15,3.335467977, 1.140190557, 1.055652435, 1.590017098, 3.188735797, 0, 0]]
X_HRJ = [[233.15,3.271797853, 1.245511112, 1.393100123, 2.148655347, 3.720826794, 0, 0.002749428],[253.15,3.271797853, 1.245511112, 1.393100123, 2.148655347, 3.720826794, 0, 0],[313.15,3.271797853, 1.245511112, 1.393100123, 2.148655347, 3.720826794, 0, 0]]


X_jetA = np.array(X_jetA)
X_HRJ = np.array(X_HRJ)
X_jetfuel = np.vstack((X_jetA,X_HRJ))

#flip colunms to match the order of the model remember the model was trained in ['T','CH3','CH','Quaternary_C','Benzyl_Rings','CH2','CH2_Chains','Cyclo_CH2']
X_jetfuel = X_jetfuel[:,[0,5,1,6,4,2,3,7]]

Y=[9.55,4.70,1.80,14.00,6.10,1.50]
T=[273.15-40,273.15-20,40+273.15,233.15,253.15,313.15]
Y = np.array(Y)
#make predictions
Y_pred = model.predict(X_jetfuel)

Y_pred = np.exp(Y_pred)

#MAPE
MAPE = np.mean(np.abs((Y - Y_pred) / Y)) * 100
print(f'MAPE: {MAPE:.2f}%')

def fit_exp_curve(T,Y):
    """
    Fits a curve of the form exp(A + B/T ) to the given 1D list of data.

    Parameters:
    T (list or np.array): The input data to fit.
    Y (list or np.array): The output data to fit.

    Returns:
    float: B coefficient.
    float: A coefficient.
    """
    #fit a curve of the form exp(A + B/T )
    T = np.array(T).reshape(-1,1)
    Y = np.log(Y)
    model = LinearRegression()
    model.fit(1/T,Y)
    A = model.intercept_
    B = model.coef_[0]
    return B,A

#fit a curve of the form exp(A + B/T ) to each 3
B_1,A_1 = fit_exp_curve(T[0:3],Y[0:3])
B_2,A_2 = fit_exp_curve(T[3:],Y[3:])
Y_pred[0] = np.exp(A_1 + B_1/T[0])
Y_pred[1] = np.exp(A_1 + B_1/T[1])
Y_pred[2] = np.exp(A_1 + B_1/T[2])

Y_pred[-1] = np.exp(A_2 + B_2/T[-1])
Y_pred[-2] = np.exp(A_2 + B_2/T[-2])
Y_pred[-3] = np.exp(A_2 + B_2/T[-3])


#plot
plt.figure()
#plot first 3
plt.plot(T[0:3],Y[0:3],label='True Jet-A Viscosity',linestyle='--')
plt.scatter(T[0:3],Y_pred[0:3],label='Predicted')
#plot last 3
plt.plot(T[3:],Y[3:],label='True HRJ Viscosity',linestyle='--')
plt.scatter(T[3:],Y_pred[3:],label='Predicted')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (mPa.s)')
plt.legend()
plt.title('Prediction of Real Fuels')
plt.show()