import numpy as np
import matplotlib.pyplot as plt
import argparse as py

import os

import parser
import converter
import utils
import time
import SVM


def visualize(X_reduced, Y, n_classes):
    class_data = []

    Y_sorted_indices = Y.argsort()

    X_sorted = X_reduced[Y_sorted_indices]
    Y_sorted = Y[Y_sorted_indices]

    k = 0

    for i in range(n_classes):
        position = np.searchsorted(Y_sorted, i, side='right')

        class_data.append(X_sorted[k:position])

        k = position

    colors = ['r', 'b']

    for i in range(n_classes):

        plt.scatter(class_data[i][:,0], class_data[i][:,1], color=colors[i])

    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('LDA')
    plt.legend()
    plt.show()

def main():

    args = parser.get_parser().parse_args()

    dataset_path = args.dataset_path
    labels_path = args.labels_path
    dataset_type = args.dataset_type
    n_classes = args.n_classes
    train_split = args.train_split

    c_param = args.c_param
    kernel = args.kernel

    dataset_name = dataset_path.strip().split('/')[-1][:-4]

    X = np.zeros(1)
    Y = np.zeros(1)

    visualise = False

    if args.visualise == 'True':
        visualise = True

    labels_path = args.labels_path

    combined = False

    if args.combined == 'True':
        combined = True

    if dataset_type == 'txt':

        if combined:
            converter.convert(dataset_path, combined)

        else:
            converter.convert(dataset_path, combined, labels_path=labels_path)

        X = np.load('datasets_npy/' + dataset_name + '.npy')
        Y = np.load('datasets_npy/' + dataset_name + '_labels' + '.npy')

    elif dataset_type == 'npy':

        if combined:

            data = np.load(dataset_path)

            X = data[:, 0:-1]

            Y = data[:, -1]

        else:

            X = np.load(dataset_path)

            Y = np.load(labels_path)


    X_train, X_test, Y_train, Y_test = utils.train_test_split(X, Y, train_split)

    dataset_name_sans_extension = dataset_name

    if not os.path.exists('datasets_npy/' + dataset_name_sans_extension):
        os.makedirs('datasets_npy/' + dataset_name_sans_extension)

    np.save('datasets_npy/' + dataset_name_sans_extension + '/' + dataset_name_sans_extension + '_train', X_train)
    np.save('datasets_npy/' + dataset_name_sans_extension + '/' + dataset_name_sans_extension + '_test', X_test)

    np.save('datasets_npy/' + dataset_name_sans_extension + '/' + dataset_name_sans_extension + '_labels_train', Y_train)
    np.save('datasets_npy/' + dataset_name_sans_extension + '/' + dataset_name_sans_extension + '_labels_test', Y_test)


    mysvm = SVM.SVM({'X_train': X_train, 'Y_train': Y_train, 'kernel': 'gaussian_kernel', 'c_param': c_param})

    mysvm.fit()

    y_predict = mysvm.predict(X_test)
    Y_test = np.where(Y_test == 0, -1 * np.ones_like(Y_test), Y_test)
    correct = np.sum(y_predict == Y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    if(X_train.shape[0]==2):
        plot_contour(X_train[y_train==1], X_train[y_train==-1], mysvm)
  

time_start = time.time();
main()
print(time.time() - time_start)
