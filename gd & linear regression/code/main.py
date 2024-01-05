#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from optimizers import GradientDescent, GradientDescentLineSearch
from fun_obj import LeastSquaresLoss, RobustRegressionLoss
from linear_models import (
    LeastSquares,
    LeastSquaresBias,
    LeastSquaresPoly,
    WeightedLeastSquares,
    LinearModel,
)
from utils import load_dataset, test_and_plot, handle, run, main


@handle("2")
def q2():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Fit least-squares estimator
    model = LeastSquares()
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model, X, y, title="Least Squares", filename="least_squares_outliers.pdf"
    )


@handle("2.1")
def q2_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    first_elements = np.ones(400)
    last_elements = 0.1 * np.ones(100)

    v = np.concatenate((first_elements, last_elements))
    
    model = WeightedLeastSquares()
    model.fit(X, y, v)

    print(model.w)

    test_and_plot(
        model, X, y, title="Weighted Least Squares", filename="w_least_squares_outliers.pdf"
    )


@handle("2.4")
def q2_4():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model,
        X,
        y,
        title="Linear Regression with Gradient Descent",
        filename="least_squares_gd.pdf",
    )


@handle("2.4.1")
def q2_4_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = RobustRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model,
        X,
        y,
        title="Linear Regression with Gradient Descent (smooth approx)",
        filename="least_squares_robust.pdf",
    )

@handle("2.4.2")
def q2_4_2():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Produce the learning curves with
    # 1. GradientDescent
    # 2. GradientDescentLineSearch

    fun_obj = RobustRegressionLoss()
    optimizer = GradientDescent(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    f_GD = np.asarray(model.fs)


    optimizer2 = GradientDescentLineSearch(max_evals=100, verbose=False)
    model2 = LinearModel(fun_obj, optimizer2)
    model2.fit(X, y)
    print(model.w)

    f_LS = np.asarray(model2.fs)

    plt.plot(f_GD, label = "Gradient Descent")
    plt.plot(f_LS, label = "GD Line Search")
    plt.title("LearningCurve")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Val")
    fname = Path("..", "figs","LearningCurve.pdf")
    plt.savefig(fname)

@handle("3")
def q3():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    # Fit least-squares model
    model = LeastSquares()
    model.fit(X, y)

    test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, no bias",
        filename="least_squares_no_bias.pdf",
    )


@handle("3.1")
def q3_1():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)
    
    model = LeastSquaresBias()
    model.fit(X, y)

    test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, yes bias",
        filename="least_squares_yes_bias.pdf",
    )


@handle("3.2")
def q3_2():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    p_vals = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100]
    num_runs = len(p_vals)
    err_trains = np.zeros(num_runs)
    err_valids = np.zeros(num_runs)

    plot_grid_size1 = int(np.ceil(np.sqrt(num_runs)))
    plot_grid_size2 = int(np.ceil(num_runs / plot_grid_size1))

    fig, axes = plt.subplots(
        plot_grid_size1,
        plot_grid_size2,
        figsize=(30, 20),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for i, (p, ax) in enumerate(zip(p_vals, (ax for row in axes for ax in row))):
        print(f"p = {p}")

        model = LeastSquaresPoly(p)
        model.fit(X, y)
        y_hat = model.predict(X)
        
        err_train = np.mean((y_hat - y) ** 2)
        err_trains[i] = err_train

        y_hat = model.predict(X_valid)
        err_valid = np.mean((y_hat - y_valid) ** 2)
        err_valids[i] = err_valid

        ax.scatter(X, y, color="b", s=2)
        Xgrid = np.linspace(np.min(X_valid), np.max(X_valid), 1000)[:, None]
        ygrid = model.predict(Xgrid)
        ax.plot(Xgrid, ygrid, color="r")
        ax.set_title(f"p={p}")
        ax.set_ylim(np.min(y), np.max(y))

    filename = Path("..", "figs", "polynomial_fits.pdf")
    print("Saving to", filename)
    fig.savefig(filename)

    # Plot error curves
    plt.figure()
    plt.plot(p_vals, err_trains, marker="o", label="training error")
    plt.plot(p_vals, err_valids, marker="o", label="validation error")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    filename = Path("..", "figs", "polynomial_error_curves.pdf")
    print("Saving to", filename)
    plt.savefig(filename)


if __name__ == "__main__":
    main()
