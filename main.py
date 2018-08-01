import pandas as pd
from numpy import array, reshape
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from demo import NeuralNetwork
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def svm_eval(x_std, labels):
    print x_std.shape
    print labels.shape
    x_train, x_test, y_train, y_test = train_test_split(
        x_std, labels, test_size=0.2, random_state=0)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc_score = accuracy_score(y_test, pred)
    print('Accuracy for SVM is: ', acc_score)

if __name__ == "__main__":
    df = pd.read_csv("Speed Dating Data.csv")
    labels = df.loc[:, "match"].values.T
    # labels = reshape(labels, (-1, 8378)).T
    df = df.drop(columns='match')
    print "Number of features {}".format(df.shape[1])
    column_with_nan = df.isnull().sum()
    # drop columns with NaN values greater than 3000
    df = df.loc[:, column_with_nan <= 3000]
    # fill missing values with mean column values
    df = df.fillna(df.mean())
    # Above results in 4 fields with still NaN values
    Nan_columns = df.columns[df.isnull().any()].tolist()
    print "Columns with NaN values left are after removing features with Nan greater than 3000 :{}".format(Nan_columns)
    df = df.drop(columns=Nan_columns)
    df = df.apply(LabelEncoder().fit_transform)
    print "Number of features left {}".format(df.shape[1])
    x = df.values
    standard_scaler = StandardScaler()
    x_std = standard_scaler.fit_transform(x)
    pca = PCA(n_components=None)
    pca.fit(x_std)
    # uncomment lines below to see variance retained vs the number of
    # components
    plt.plot(range(0, 119), pca.explained_variance_ratio_)
    plt.show()
    pca = PCA(n_components=85)
    x_std = pca.fit_transform(x_std)
    # around 94 percent variance is retained
    print "Variance retained {}".format(sum(pca.explained_variance_ratio_))
    # neural_network = NeuralNetwork()
    # neural_network.train(x_std,
    #                      labels.T, 100, 0.01)
    svm_eval(x_std, labels)
