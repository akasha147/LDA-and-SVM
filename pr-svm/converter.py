import numpy as np

def convert(dataset_path, combined, labels_path=''):
    data = np.loadtxt(dataset_path,dtype=np.float32, delimiter=',')

    if(combined):

        print(data.shape)

        X = data[:,0:-1]

        Y = data[:,-1]

        print(X)

        print(Y)

        dataset_name = dataset_path.strip().split('/')
        dataset_name = dataset_name[-1]
        dataset_name = dataset_name[0:-4]

        np.save('datasets_npy/' + dataset_name, X)
        np.save('datasets_npy/' + dataset_name + '_labels', Y)

    else:
        labels = np.loadtxt(labels_path, dtype=np.float32, delimiter=',')

        X = data

        Y = labels

        print(X)

        print(Y)

        dataset_name = dataset_path.strip().split('/')
        dataset_name = dataset_name[-1]
        dataset_name = dataset_name[0:-4]

        np.save('datasets_npy/' + dataset_name, X)
        np.save('datasets_npy/' + dataset_name + '_labels', Y)
