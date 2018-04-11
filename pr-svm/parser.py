import argparse

parser = argparse.ArgumentParser(description='process arguments for LDA')

parser.add_argument('--dataset_path', default='data_banknote_authentication.txt', help='path to dataset to be used')
parser.add_argument('--labels_path', default='data_banknote_authentication_labels.npy', help='path to labels if separate')
parser.add_argument('--dataset_type', default='txt', help="numpy array or txt file ? numpy or txt")
parser.add_argument('--visualise', default='True', help='visalise data ?')
parser.add_argument('--combined', default='True', help='are training data and labels combined ?')
parser.add_argument('--n_classes', default='2', type=int, help='no of classes')
parser.add_argument('--train_split', default='0.70', type=float, help='test split for data ?')

parser.add_argument('--c_param', default='100.00', type=float, help='c parameter ?')
parser.add_argument('--kernel', default='dot_product', help='kernel function to use. options are : dot_product, ')

def get_parser():
    return parser