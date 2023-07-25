import pandas as pd
import numpy as np
import sys
import os










def main():
    # params
    data_path = r'D:\denis'
    path_to_train_labels= os.path.join(data_path, 'train.csv')
    path_to_dev_labels= os.path.join(data_path, 'dev.csv')
    path_to_test_labels= os.path.join(data_path, 'test.csv')
    batch_size = 32
    # load labels
    train_labels = pd.read_csv(path_to_train_labels)
    dev_labels = pd.read_csv(path_to_dev_labels)
    test_labels = pd.read_csv(path_to_test_labels)




if __name__=="__main__":
    main()