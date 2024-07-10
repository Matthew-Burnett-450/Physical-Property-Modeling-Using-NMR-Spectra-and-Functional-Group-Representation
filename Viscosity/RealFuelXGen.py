import numpy as np
import matplotlib.pyplot as plt
import json
import gtda.diagrams as diag
from scipy.stats import skew, kurtosis


#fuels = {
#    'Jet-A POSF 10325': [[273.15-40,273.15-20,40+273.15],[9.55,4.70,1.80]],
#    'HRJ POSF 7720': [[233.15,253.15,313.15],[14.00,6.10,1.50]],
#}

Y=[[9.55,4.70,1.80],[14.00,6.10,1.50]]
T=[[273.15-40,273.15-20,40+273.15],[233.15,253.15,313.15]]
idx=[0,0,0,1,1,1]
#load 7720 HRJ.csv and 10325 Jet A.csv colums are shifts, intensities
data_7720 = np.genfromtxt('7720 HRJ.csv', delimiter=',',skip_header=1)
data_10325 = np.genfromtxt('10325 Jet A.csv', delimiter=',',skip_header=1)

#remove negative shifts
data_7720 = data_7720[data_7720[:,0]>0]
data_10325 = data_10325[data_10325[:,0]>0]
#remove zero intensities
data_7720 = data_7720[data_7720[:,1]>0]
data_10325 = data_10325[data_10325[:,1]>0]



#put onto histogram to make point cloud
hist, bin_edges = np.histogram(data_7720[:,0], weights=data_7720[:,1], bins=250,range=(0,250),density=True)
temp_vec = []
for i in range(len(hist)):
    x_coord = i
    y_coord = hist[i]
    temp_vec.append([x_coord,y_coord,1])
data_7720 = np.array(temp_vec)

hist, bin_edges = np.histogram(data_10325[:,0], weights=data_10325[:,1], bins=250,range=(0,250),density=True)
temp_vec = []
for i in range(len(hist)):
    x_coord = i
    y_coord = hist[i]
    temp_vec.append([x_coord,y_coord,1])
data_10325 = np.array(temp_vec)

plt.scatter(data_7720[:,0],data_7720[:,1])
plt.show()

X_point_cloud = np.array([data_7720,data_10325])


#plot point cloud
for i in range(len(X_point_cloud)):
    plt.plot(X_point_cloud[i][:,0],X_point_cloud[i][:,1])
    plt.show()


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

    X_stats_new=[]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            X_stats_new.append(X_stats[i])
    X_stats = np.array(X_stats_new)

    X = np.hstack((X,X_stats))
    np.save('X_jetfuels.npy',X)

    T = [item for sublist in T for item in sublist]
    Y = [item for sublist in Y for item in sublist]

    Y = np.array(Y)
    np.save('Y_jetfuels.npy',Y)
    T = np.array(T)
    np.save('T_jetfuels.npy',T)
    idx = np.array(idx)
    np.save('idx_jetfuels.npy',idx)
    #print shapes
    print(X.shape)
    print(Y.shape)
    print(T.shape)
    print(idx.shape)