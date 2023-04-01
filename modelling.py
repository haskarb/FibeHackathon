import pandas as pd
import numpy as np


def basic_preprocess(X_train, X_test):

    # Drop id columns
    X_train.drop(['id'], axis=1, inplace=True)
    X_test.drop(['id'], axis=1, inplace=True)


    # Drop columns with null values > 50%
    null_cols = X_train.isnull().mean()
    null_cols = null_cols[null_cols > 0.5]
    X_train.drop(null_cols.index, axis=1, inplace=True)
    X_test.drop(null_cols.index, axis=1, inplace=True)


    # Drop columns with constant values
    const_cols = X_train.nunique()
    const_cols = const_cols[const_cols == 1]
    X_train.drop(const_cols.index, axis=1, inplace=True)
    X_test.drop(const_cols.index, axis=1, inplace=True)


    # Drop columns with duplicated values
    dup_cols = X_train.T.duplicated()
    dup_cols = dup_cols[dup_cols == True]
    X_train.drop(dup_cols.index, axis=1, inplace=True)
    X_test.drop(dup_cols.index, axis=1, inplace=True)


    return X_train, X_test

def model(X_train, y_train):
    pass

def submission_file(y_pred):
    pass



if __name__ == '__main__':

    Xy_train = pd.read_csv('dataset/train.csv')
    X_train = Xy_train.drop('target', axis=1)
    y_train = Xy_train['target']

    X_test = pd.read_csv('dataset/test.csv')

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    print('Preprocessing data...')
    X_train, X_test = basic_preprocess(X_train, X_test)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))