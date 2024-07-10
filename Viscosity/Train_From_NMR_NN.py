import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from Util import load_data, fit_fourier_series,cutoff_point_detector
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVR
import gtda.diagrams as diag
from scipy.stats import skew, kurtosis
import sweetviz as sv
import torch
from Model import MLPModel
# Load data
X_point_cloud, FG, T, Y, FG_labels,idx = load_data()

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
    hist, bin_edges = np.histogram(x, bins=250)
    #if not zero set to 1
    hist
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
        print(i//len(X_point_cloud)*100)

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



plt.scatter(T,Y)
plt.show()

#save X
X = np.array(X)
Y = np.array(Y)
T = np.array(T)
idx = np.array(idx)

T = T.reshape(-1,1)
X = np.hstack((X,T))

Y=np.log(Y)

plt.scatter(T,Y)
plt.show()

"""#sweetviz X,Y
df=pd.DataFrame(X,columns=['bottleneck','persistence_image','mean','max','min','std','average','skew2',"range",'T'])
df['Y']=Y
report = sv.analyze(df)
report.show_html('sweetviz.html')"""


print(X.shape)
print(Y.shape)


# Split data into training and testing sets
train_inds, test_inds = next(GroupShuffleSplit(test_size=0.2, n_splits=2,random_state=42).split(X, groups=idx))

X_train, X_test = X[train_inds], X[test_inds]
Y_train, Y_test = Y[train_inds], Y[test_inds] 

#convert to torch
X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).float()
X_test = torch.tensor(X_test).float()
Y_test = torch.tensor(Y_test).float()

# Define the model
model = MLPModel(input_dim=X_train.shape[1], output_dim=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(50):
    optimizer.zero_grad()
    Y_pred = model(X_train)
    loss = criterion(Y_pred, Y_train)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Epoch {i} loss: {loss.item()}')

# Predict on the test set
Y_pred = model(X_test).detach().numpy()
Y_train_pred = model(X_train).detach().numpy()
Y_test = Y_test.detach().numpy()
Y_train = Y_train.detach().numpy()


Y_pred = np.exp(Y_pred)
Y_train_pred = np.exp(Y_train_pred)
Y_test = np.exp(Y_test)
Y_train = np.exp(Y_train)

MAPE = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
MSE = np.mean((Y_test - Y_pred)**2)
print(f'MSE: {MSE:.2f}')
print(f'MAPE: {MAPE:.2f}%')
print(f'Data Size: {max(idx)}')  
#parity plot
plt.scatter(Y_test,Y_pred,c='blue',s=5,label='Test')
plt.scatter(Y_train,Y_train_pred,c='orange',s=5,label='Train')
plt.plot([0, 1], [0, 1],ls="--", c='k')
plt.xlim([0,max(Y_test)])
plt.ylim([0,max(Y_test)])
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Parity Plot')
plt.legend()
plt.show()



