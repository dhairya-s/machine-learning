import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                y_yes_mode = utils.mode(y[X[:, j] > t])
                y_no_mode = utils.mode(y[X[:,j] <= t])
                

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp) 

def calc_p(bincount):
    output = []
    total = 0
    for i in bincount:
        total += i
    
    for i in bincount:
        output.append(i/total)
    
    return np.array(output)
class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape
        
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)
        # print(count)
        # print("here")
        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # minError = np.sum(y != y_mode)
        # p_y = np.array([count[0]/(count[0]+count[1]), count[1]/(count[0]+count[1])])
        p_y = calc_p(count)
        minlen_bincount = count.shape[0]
        maxInfoGain = 0
        
        # print('maxInfoGain ',maxInfoGain)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                y_above = y[X[:, j] > t]
                y_below = y[X[:,j] <= t]
                y_yes_mode = utils.mode(y_above)
                y_no_mode = utils.mode(y_below)

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= t] = y_no_mode

                # Compute error
                # errors = np.sum(y_pred != y)
                # p_y_above = np.array([np.sum(y_above==1)/len(y_above), (len(y_above)-np.sum(y_above==1))/len(y_above)])
                # p_y_below = np.array([np.sum(y_below==1)/len(y_below), (len(y_below)-np.sum(y_below==1))/len(y_below)])
                above_count = np.bincount(y_above, minlength=minlen_bincount)
                below_count = np.bincount(y_below, minlength=minlen_bincount) 

                # print(above_count)
                # p_y_above = np.array([(above_count[0]/(above_count[0]+above_count[1])), (above_count[1]/(above_count[0]+above_count[1]))])
                p_y_above = calc_p(above_count)
                p_y_below = calc_p(below_count)
                # p_y_below = np.array([below_count[0]/(below_count[0]+below_count[1]), below_count[1]/(below_count[0]+below_count[1])])
                # Computer Info Gain
                infoGain = entropy(p_y) - (len(y_above)/len(y))*entropy(p_y_above) - (len(y_below)/len(y))*entropy(p_y_below)
                # print('infoGain', infoGain)
                # Compare to minimum error so far
                if maxInfoGain < infoGain:
                    # This is the lowest error, store this value
                    maxInfoGain = infoGain
                    # print('maxInfoGain ',maxInfoGain)
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

def hardcoded_predict(sample):
    if  sample[0] > -80.305106:
        if sample[1] > 36.453576:
            return 0
        else:
            return 1
    else:
        if sample[1]>37.669007:
            return 0
        else:
            return 1
