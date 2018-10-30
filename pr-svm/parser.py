import argparse

parser = argparse.ArgumentParser(description='Process arguments for LDA')

parser.add_argument('--dataset_path', default='data_banknote_authentication.txt', help='Path to dataset file')
parser.add_argument('--labels_path', default='data_banknote_authentication_labels.npy', help='Path to labels file')
parser.add_argument('--dataset_type', default='txt', help="numpy array file or text file ? numpy/txt")
parser.add_argument('--visualise', default='True', help='Visalise data ?')
parser.add_argument('--combined', default='True', help='Are training data and labels combined ?')
parser.add_argument('--n_classes', default='2', type=int, help='No. of classes')
parser.add_argument('--train_split', default='0.70', type=float, help='Test split for data ?')

parser.add_argument('--c_param', default='100.00', type=float, help='C Parameter value')
parser.add_argument('--kernel', default='dot_product', help='Kernel function to use. Options are : dot_product, ')

def get_parser():
    return parser
