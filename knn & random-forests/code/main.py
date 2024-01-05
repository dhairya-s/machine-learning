#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    k_list = [1,3,10]

    training_err = np.zeros(3)
    test_err = np.zeros(3)

    for i, k in enumerate(k_list):
        model = KNN(k)
        model.fit(X,y)

        y_pred_train = model.predict(X)
        training_err[i] = np.mean(y_pred_train != y)

        y_pred_test = model.predict(X_test)
        test_err[i] = np.mean(y_pred_test != y_test)
        

    print("Training error", training_err)
    print("Test error", test_err)

    model = KNN(1)
    model.fit(X,y)
    plot_classifier(model, X, y)
    fname = Path("..", "figs", "q3_knn.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

    sk_model = KNeighborsClassifier(n_neighbors=3);
    sk_model.fit(X,y)
    plot_classifier(sk_model, X, y)
    sk_fname = Path("..", "figs", "q3_sklearn_knn.pdf")
    plt.savefig(sk_fname)
    print("\nFigure saved as '%s'" % sk_fname)



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    cv_accs_seperate = np.zeros((10,len(ks)))
    
    total_ele_x = X.size
    total_ele_y = y.size
    ten_percent_validate = 0.1*total_ele_x
    ten_percent_validate_y = 0.1*total_ele_y

    for i in range(10):
        mask = np.zeros(X.shape, dtype=bool)
        mask_y = np.zeros(y.shape, dtype=bool)
        train_mask = mask.ravel()
        train_mask_y = mask_y.ravel()

        train_mask[int(i*ten_percent_validate):int((i+1)*ten_percent_validate)] = True 
        validate_mask = ~train_mask

        train_mask_y[int(i*ten_percent_validate_y):int((i+1)*ten_percent_validate_y)] = True 
        validate_mask_y = ~train_mask_y

        train_mask = train_mask.reshape(mask.shape)
        validate_mask = validate_mask.reshape(mask.shape)

        train_mask_y = train_mask_y.reshape(mask_y.shape)
        validate_mask_y = validate_mask_y.reshape(mask_y.shape)

        for j,k in enumerate(ks):
            model = KNeighborsClassifier(n_neighbors=k)
            fold_train_set = np.ma.array(X, mask=train_mask)
            fold_validate_set = np.ma.array(X, mask=validate_mask)

            fold_train_set_y = np.ma.array(y, mask=train_mask_y)
            fold_validate_set_y = np.ma.array(y, mask=validate_mask_y)

            model.fit(fold_train_set,fold_train_set_y)

            y_pred_train = model.predict(fold_validate_set)
            cv_accs_seperate[i][j] = np.mean(y_pred_train != fold_validate_set_y)
    
    cv_accs = np.mean(cv_accs_seperate, axis=0)
    print(cv_accs)

    training_err = np.zeros(len(ks))
    test_err = np.zeros(len(ks))
    for i,k in enumerate(ks):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X,y)

        y_pred_train = model.predict(X)
        training_err[i] = np.mean(y_pred_train != y)

        y_pred_test = model.predict(X_test)
        test_err[i] = np.mean(y_pred_test != y_test)
        print(k)
    
    print(test_err)
    plt.plot(ks, test_err, label="test-error")
    plt.plot(ks, cv_accs, label="cross-validation")

    plt.figure(1)
    plt.xlabel("k")
    plt.ylabel("Classification error")
    plt.legend()
    fname = Path("..", "figs", "q2_2_errors.pdf")
    plt.savefig(fname)
    
    plt.figure(2)
    plt.plot(ks, training_err, label="training-error")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend()
    fname = Path("..", "figs", "q2_4_errors.pdf")
    plt.savefig(fname)


    
    print("\nFigure saved as '%s'" % fname)


    




@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    raise NotImplementedError()



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    ran = 50
    errors = np.zeros(ran) # run 50 times
    models = []

    for i in range(ran):
        model = Kmeans(k=4)
        model.fit(X)
        
        models.append(model)
        errors[i] = model.error(X, model.y, model.means)

    lowest_err_model = models[np.argmin(errors)]
    print(errors[np.argmin(errors)]) #3071.468052653851

    y = lowest_err_model.predict(X)    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_best_of_50.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]
    ks = np.arange(1, 11)
    best_errors = np.zeros(len(ks))

    for k in ks:
        errors = np.zeros(50)
        models = []
        for i in range(50):
            model = Kmeans(k=k)
            model.fit(X)

            models.append(model)
            errors[i] = model.error(X, model.y, model.means)

        best_model = models[np.argmin(errors)]
        best_errors[k - 1] = best_model.error(X, best_model.y, best_model.means)


    plt.xlabel("k")
    plt.ylabel("Error")

    plt.plot(ks, best_errors, label="Error")
    fname = os.path.join("..", "figs", "q5_2_error_k.png")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)



if __name__ == "__main__":
    main()
