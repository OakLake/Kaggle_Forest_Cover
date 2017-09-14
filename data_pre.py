# data preprocessor

import csv
import numpy as np
from six.moves import cPickle as pickle
import os

os.system('clear')

file_name = './test.csv' # adjust line 42

with open(file_name) as f:
    read = csv.reader(f,delimiter = ',')
    _ = next(read) # get rid of first line containing field names
    temp_dataset = []
    for line in read:
        example = [float(i) for i in line]
        temp_dataset.append(example)

    dataset = np.array(temp_dataset)

def normalise_data(data,columns):

    mean = np.mean(data,axis = 0)
    std  = np.std(data,axis = 0)

    temp_data = np.copy(data)

    for col in columns:
        temp_data[:,col] -=  mean[col]
        temp_data[:,col] /= std[col]

    return temp_data


norm_data = normalise_data(dataset,[1,2,3,4,5,6,7,8,9,10])

# print(dataset[:1])
# print(norm_data[:1])


save_file = './test.pickle'
save_data = norm_data[:,1:]

with open(save_file,'wb') as f:
    pickle.dump(save_data,f,protocol= pickle.HIGHEST_PROTOCOL)

print('Pickle Complete!')
