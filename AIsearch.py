import numpy 
import sys 
import nmslib 
import time 
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
print(sys.version)
print("NMSLIB version:", nmslib.__version__)


# Just read the data
all_data_matrix = pd.read_csv("TimeBasedFeatures-10s-Layer2.csv")
df = all_data_matrix.drop('Source IP', axis=1)
# df = all_data_matrix.drop('Source Port', axis=1, inplace=True)
# label = df[['label']]
# df = df.drop('label', axis=1)
df = df.drop(' Destination IP', axis=1)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(df.label)
df['label'] = le.transform(df.label)

all_data_matrix = df.to_numpy()
# all_data_matrix = numpy.loadtxt(open("TBF.csv", "rb"), delimiter=",", skiprows=1)
# numpy.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1)

# Create a held-out query data set
(data_matrix, query_matrix) = train_test_split(all_data_matrix, test_size = 0.1)

print("# of queries %d, # of data points %d"  % (query_matrix.shape[0], data_matrix.shape[0]) )

# Set index parameters
# These are the most important onese
NN = 15
efC = 100

num_threads = 4
index_time_params = {'NN': NN, 'indexThreadQty': num_threads, 'efConstruction': efC}

# Number of neighbors
K=100

# Space name should correspond to the space name
# used for brute-force search
space_name='kldivgenfast'


# Intitialize the library, specify the space, the type of the vector and add data points
index = nmslib.init(method='sw-graph', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
index.addDataPointBatch(data_matrix)


# Create an index
start = time.time()
index.createIndex(index_time_params)
end = time.time()
print('Index-time parameters', index_time_params)
print('Indexing time = %f' % (end-start))

# Setting query-time parameters
efS = 100
query_time_params = {'efSearch': efS}
print('Setting query-time parameters', query_time_params)
index.setQueryTimeParams(query_time_params)


# Querying
query_qty = query_matrix.shape[0]
start = time.time()
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end = time.time()
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
      (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty))

# Computing gold-standard data
print('Computing gold-standard data')

start = time.time()

gs = []

query_qty = query_matrix.shape[0]
data_qty = data_matrix.shape[0]

for i in range(query_qty):
    q = query_matrix[i, :]
    d = numpy.log(data_matrix * (1.0 / q))
    dist_vals = numpy.sum(data_matrix * d, axis=1)
    tmp = [(dist_vals[i], i) for i in range(data_qty)]
    tmp.sort()
    gs.append([tmp[i][1] for i in range(K)])

end = time.time()

print('brute-force kNN time total=%f (sec), per query=%f (sec)' %
      (end - start, float(end - start) / query_qty))

# Finally computing recall
recall=0.0
for i in range(0, query_qty):
  correct_set = set(gs[i])
  ret_set = set(nbrs[i][0])
  recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
recall = recall / query_qty
print('kNN recall %f' % recall)


# Save a meta index and the data
index.saveIndex('dense_index_kldiv.txt', save_data=True)


# Re-intitialize the library, specify the space, the type of the vector.
newIndex = nmslib.init(method='sw-graph', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)


# Re-load the index and the data
newIndex.loadIndex('dense_index_kldiv.txt', load_data=True)


# Setting query-time parameters and querying
print('Setting query-time parameters', query_time_params)
newIndex.setQueryTimeParams(query_time_params)

query_qty = query_matrix.shape[0]
start = time.time()
new_nbrs = newIndex.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end = time.time()
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
      (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty))



# Finally computing recall
recall=0.0
for i in range(0, query_qty):
  correct_set = set(gs[i])
  ret_set = set(nbrs[i][0])
  recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
recall = recall / query_qty
print('kNN recall %f' % recall)
