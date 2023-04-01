import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess(X_train, X_test):
    # Drop id columns
    X_train.drop(["id"], axis=1, inplace=True)
    X_test.drop(["id"], axis=1, inplace=True)

    # Drop columns with constant values
    const_cols = X_train.nunique()
    const_cols = const_cols[const_cols == 1]
    X_train.drop(const_cols.index, axis=1, inplace=True)
    X_test.drop(const_cols.index, axis=1, inplace=True)

    # Drop columns with null values > 50%
    null_cols = X_train.isnull().mean()
    null_cols = null_cols[null_cols > 0.5]
    X_train.drop(null_cols.index, axis=1, inplace=True)
    X_test.drop(null_cols.index, axis=1, inplace=True)

    # impute missing values with the mean
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    # Drop columns with duplicated values
    dup_cols = X_train.T.duplicated()
    dup_cols = dup_cols[dup_cols == True]
    X_train.drop(dup_cols.index, axis=1, inplace=True)
    X_test.drop(dup_cols.index, axis=1, inplace=True)

    # drop columns with inf values
    inf_cols = X_train.isin([np.inf, -np.inf]).sum()
    inf_cols = inf_cols[inf_cols > 0]
    X_train.drop(inf_cols.index, axis=1, inplace=True)
    X_test.drop(inf_cols.index, axis=1, inplace=True)

    return X_train, X_test


def model(X_train, y_train):
    # split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # scale the data
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # create a logistic regression model
    clf = LogisticRegressionCV(
        cv=3,
        solver="newton-cholesky",
        max_iter=1000,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        class_weight="balanced",
    )

    # fit the model
    clf.fit(X_train, y_train)

    # predict the validation set
    y_pred = clf.predict(X_val)

    # calculate the f1 score, accuracy and auc
    f1 = f1_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)

    print("f1 score: {}".format(f1))
    print("accuracy: {}".format(acc))
    print("auc: {}".format(auc))

    return clf


def submission_file(y_pred):
    pass


if __name__ == "__main__":
    Xy_train = pd.read_csv("dataset/train.csv")
    X_train = Xy_train.drop("target", axis=1)
    y_train = Xy_train["target"]

    X_test = pd.read_csv("dataset/test.csv")

    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))

    print("Preprocessing data...")
    X_train, X_test = preprocess(X_train, X_test)

    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))

    print("Training model...")
    clf = model(X_train, y_train)

    # check null values
    # print(X_train.isnull().sum())
