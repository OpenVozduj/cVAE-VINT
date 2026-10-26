import numpy as np
from sklearn.preprocessing import MinMaxScaler


db2b = np.load('database2b.npy')
db3b = np.load('database3b.npy')
db4b = np.load('database4b.npy')

dbG = np.concatenate((db2b, db3b, db4b), axis=0)

dataBC = dbG[:,[13, 14, 16, 19]]
dataBX = np.delete(dbG, [13, 14, 16, 19], axis=1)

scalerX = MinMaxScaler(feature_range=(0, 1))
dataBX_scaled = scalerX.fit_transform(dataBX)

scalerC = MinMaxScaler(feature_range=(0, 1))
dataBC_scaled = scalerC.fit_transform(dataBC)

#%% General database
np.save('dataBX.npy', dataBX_scaled)
np.save('dataBC.npy', dataBC_scaled)

#%% 
# index_test = np.random.choice(range(len(dataBX)), size=3346, replace=False)
# index_test.sort()

# dataBX_test = dataBX_scaled[index_test]
# dataBX_train = np.delete(dataBX_scaled, index_test, axis=0)

# dataBC_test = dataBC_scaled[index_test]
# dataBC_train = np.delete(dataBC_scaled, index_test, axis=0)

# np.save('dataBX_test.npy', dataBX_test)
# np.save('dataBX_train.npy', dataBX_train)
# np.save('dataBC_test.npy', dataBC_test)
# np.save('dataBC_train.npy', dataBC_train)
