
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Testing the dataset 
# Paths de los archivos


#--------------------------------------------------------------------------------

TRAIN_PATH = r"data/TrainData/img/*.h5"
TRAIN_MASK = r'data/TrainData/mask/*.h5'

TRAIN_XX = np.zeros((3799, 128, 128, 9))
TRAIN_YY = np.zeros((3799, 128, 128, 1))
all_train = sorted(glob.glob(TRAIN_PATH))
all_mask = sorted(glob.glob(TRAIN_MASK))

for i, (img, mask) in enumerate(zip(all_train, all_mask)):
    print(i, img, mask)
    with h5py.File(img) as hdf:
        ls = list(hdf.keys())
        data = np.array(hdf.get('img'))

        # assign 0 for the nan value
        data[np.isnan(data)] = 0.000001

        # to normalize the data 
        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0

        # ndvi calculation
        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red,np.add(data_nir, data_red))
        
        # final array
        TRAIN_XX[i, :, :, 0] = 1-data[:, :, 3]/mid_rgb #RED
        TRAIN_XX[i, :, :, 1] =  1-data[:, :, 2] /mid_rgb #GREEN
        TRAIN_XX[i, :, :, 2] =  1-data[:, :, 1] /mid_rgb#BLUE
        TRAIN_XX[i, :, :, 3] = data_ndvi #NDVI
        TRAIN_XX[i, :, :, 4] = 1 - data[:, :, 12] / mid_slope #SLOPE
        TRAIN_XX[i, :, :, 5] = 1 - data[:, :, 13] / mid_elevation #ELEVATION
        TRAIN_XX[i, :, :, 6] = data[:, :, 3] #RAW RED 
        TRAIN_XX[i, :, :, 7] =  data[:, :, 2] #RAW GREEN
        TRAIN_XX[i, :, :, 8] =  data[:, :, 1] #RAW BLUE
    
    
    with h5py.File(mask) as hdf:
        ls = list(hdf.keys())
        data=np.array(hdf.get('mask'))
        TRAIN_YY[i, :, :, 0] = data


x_train, x_valid, y_train, y_valid = train_test_split(TRAIN_XX, TRAIN_YY, test_size=0.2, shuffle= True)

np.save('./results/x_train.npy', x_train)
np.save('./results/x_valid.npy', x_valid)
np.save('./results/y_train.npy', y_train)
np.save('./results/y_valid.npy', y_valid)