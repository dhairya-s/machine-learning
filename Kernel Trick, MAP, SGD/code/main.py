#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(lr_model, X_train, y_train)
    savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")
    kernels = [PolynomialKernel(2), GaussianRBFKernel(0.5)]

    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()

    for kernel in kernels:
        model = KernelClassifier(loss_fn, optimizer, kernel)
        model.fit(X_train, y_train)

        print(f"Training error {np.mean(model.predict(X_train) != y_train):.1%}")
        print(f"Validation error {np.mean(model.predict(X_val) != y_val):.1%}")

        fig = plot_classifier(model, X_train, y_train)
        savefig(kernel.name()+".png", fig)



@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    training_error = np.inf
    sigma_train = sigmas[0]
    lammy_train = lammys[0]

    validation_error = np.inf
    sigma_valid = sigmas[0]
    lammy_valid = lammys[0]

    for i in range(len(sigmas)):
        for j in range(len(lammys)):
                kernel = GaussianRBFKernel(sigma=sigmas[i])
                loss_fn = KernelLogisticRegressionLossL2(lammy=lammys[j])
                optimizer = GradientDescentLineSearch()

                model = KernelClassifier(loss_fn, optimizer, kernel)
                model.fit(X_train, y_train)
                
                train_errs[i,j] = np.mean(model.predict(X_train) != y_train)
                val_errs[i,j] = np.mean(model.predict(X_val) != y_val)

                if (train_errs[i,j] < training_error):
                    training_error = train_errs[i,j]
                    sigma_train = sigmas[i]
                    lammy_train = lammys[j] 
                
                if (val_errs[i,j] < validation_error):
                    validation_error = val_errs[i,j]
                    sigma_valid = sigmas[i]
                    lammy_valid = lammys[j] 

    sigmas = [sigma_train, sigma_valid]
    lammys = [lammy_train, lammy_valid]
    names = ["training", "validation"]

    for r in range(2):
        kernel = GaussianRBFKernel(sigma=sigmas[r])
        loss_fn = KernelLogisticRegressionLossL2(lammy=lammys[r])
        optimizer = GradientDescentLineSearch()

        model = KernelClassifier(loss_fn, optimizer, kernel)
        model.fit(X_train, y_train)
        
        fig = plot_classifier(model, X_train, y_train)
        savefig(names[r]+".png", fig)


    print(f"Best Training error: {training_error}")
    print(f"Sigma Training: {sigma_train}")
    print(f"Lambda Training: {lammy_train}")

    print(f"Best Validation error: {validation_error}")
    print(f"Sigma Validation: {sigma_valid}")
    print(f"Lambda Validation: {lammy_valid}")

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    """YOUR CODE HERE FOR Q3"""
    pass


@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    loss_fn = LeastSquaresLoss()
    base_optimizer = GradientDescent()
    learning_rate = ConstantLR(0.0003)
    batch_size = [1, 10, 100]
    
    for size in batch_size:
        optimizer = StochasticGradient(base_optimizer, learning_rate, batch_size=size, max_evals=10)
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train, y_train)
        print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
        print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")


@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    fig, ax = plt.subplots()
    lr_naming = ["ConstantLR", "InverseLR", "InverseSquaredLR", "InverseSqrtLR"]

    c = 0.1
    loss_fn = LeastSquaresLoss()
    base_optimizer = GradientDescent()
    learning_rate = [ConstantLR(c), InverseLR(c), InverseSquaredLR(c), InverseSqrtLR(c)]

    for i in range(4):
        optimizer = StochasticGradient(base_optimizer, learning_rate[i], batch_size=10, max_evals=50)
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train, y_train)
        ax.plot(model.fs, label=lr_naming[i])

    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    ax.legend()
    savefig("4.3.png", fig)
    


if __name__ == "__main__":
    main()
