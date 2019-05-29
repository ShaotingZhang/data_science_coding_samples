import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class TreeNode:
    def __init__(self, is_leaf, prediction, split_feature):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.left = None
        self.right = None

class WeightedDecisionTree(BaseEstimator):
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth
        self.min_error = min_error

    def fit(self, X, Y, data_weights=None):
        data_set = pd.concat([X, Y], axis=1)
        features = X.columns
        target = Y.columns[0]
        self.root_node = create_weighted_tree(data_set, data_weights, features, target, current_depth=0, max_depth=self.max_depth, min_error=self.min_error)

    def predict(self, X):
        prediction = X.apply(lambda row: predict_single_data(self.root_node, row), axis=1)
        return prediction

    def score(self, testX, testY):
        target = testY.columns[0]
        result = self.predict(testX)
        return accuracy_score(testY[target], result)

class MyAdaboost(BaseEstimator):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y):
        self.models = []
        self.model_weights = []
        self.target = Y.columns[0]

        N, _ = X.shape
        alpha = np.ones(N) / N

        for m in range(self.M):
            tree = WeightedDecisionTree(max_depth=2, min_error=1e-15)
            tree.fit(X, Y, data_weights=alpha)
            prediction = tree.predict(X)

            weighted_error = alpha.dot(prediction != Y[self.target])

            model_weight = 0.5 * (np.log(1 - weighted_error) - np.log(weighted_error))
            
            alpha = alpha * np.exp(-model_weight * Y[self.target] * prediction)
            alpha = alpha / alpha.sum()

            self.models.append(tree)
            self.model_weights.append(model_weight)

    def predict(self, X):
        N, _ = X.shape
        result = np.zeros(N)
        for wt, tree in zip(self.model_weights, self.models):
            result += wt * tree.predict(X)
        return np.sign(result)

    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY[self.target], result)


def dummies(data, columns=['pclass', 'name_title', 'embarked', 'sex']):
    """
    add the column to the data
    """
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data


def node_weighted_mistakes(targets_in_node, data_weights):
    # sum all the weights of data with the lable is +1
    weight_positive = sum(data_weights[targets_in_node == +1])
    weighted_mistakes_negative = weight_positive 
    # sum all the weights of data with the lable is -1
    weight_negative = sum(data_weights[targets_in_node == -1])
    weighted_mistakes_positive = weight_negative

    # return the mistake weight with the label
    if weighted_mistakes_negative < weighted_mistakes_positive:
        return (weighted_mistakes_negative, -1)
    else:
        return (weighted_mistakes_positive, + 1)


def best_split_weighted(data, features, target, data_weights):
    # return the best feature
    best_feature = None
    best_error = float("inf")
    num_data_points = float(len(data))

    for feature in features:
        # the data with the feature is 0 in the left split
        left_split = data[data[feature] == 0]
        left_data_weights = data_weights[data[feature] == 0]

        # the data with the feature is 1 in the right split
        right_split = data[data[feature] == 1]
        right_data_weights = data_weights[data[feature] == 1]

        # count the mistakes in the left split
        left_misses, left_class = node_weighted_mistakes(left_split[target], left_data_weights)
        # count the mistakes in the right split        
        right_misses, right_class = node_weighted_mistakes(right_split[target], right_data_weights)
        # count the error
        error = (left_misses + right_misses) * 1.0 / sum(data_weights)
        # update the feature and error, the error is lower, the feature is better
        if error < best_error:
            best_error = error
            best_feature = feature
    return best_feature

def create_leaf(target_values, data_weights):
    # initial the node
    leaf = TreeNode(True, None, None)
    # get the result of prediction of this leaf
    weighted_error, prediction_class = node_weighted_mistakes(target_values, data_weights)
    leaf.prediction = prediction_class
    return leaf


def create_weighted_tree(data, data_weights, features, target, current_depth=0, max_depth=10, min_error=1e-15):
    # copy the useful feature
    remaining_features = features[:]
    target_values = data[target]
    
    # termination 1
    if node_weighted_mistakes(target_values, data_weights)[0] <= min_error:
        print("Termination 1 reached.")
        return create_leaf(target_values, data_weights)
    # termination 2
    if len(remaining_features) == 0:
        print("Termination 2 reached.")
        return create_leaf(target_values, data_weights)
    # termination 3
    if current_depth >= max_depth:
        print("Termination 3 reached.")
        return create_leaf(target_values, data_weights)

    # choose the best features
    split_feature = best_split_weighted(data, features, target, data_weights)
    # put the features with 0 on the left and the features with 1 on the right
    left_split = data[data[split_feature] == 0]
    right_split = data[data[split_feature] == 1]
    # put the datas with weight 0 on the left and the datas with weight 1 on the right
    left_data_weights = data_weights[data[split_feature] == 0]
    right_data_weights = data_weights[data[split_feature] == 1]
    # remove the features we used
    remaining_features = remaining_features.drop(split_feature)
    print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))

    # if all datas are on one side, we just create the leaf and return
    if len(left_split) == len(data):
        print("Perfect split!")
        return create_leaf(left_split[target], left_data_weights)
    if len(right_split) == len(data):
        print("Perfect split!")
        return create_leaf(right_split[target], right_data_weights)

    # repeat the steps above with recursion
    left_tree = create_weighted_tree(left_split, left_data_weights, remaining_features, target, current_depth + 1, max_depth, min_error)
    right_tree = create_weighted_tree(right_split, right_data_weights, remaining_features, target, current_depth + 1, max_depth, min_error)

    # generate the current node
    result_node = TreeNode(False, None, split_feature)
    result_node.left = left_tree
    result_node.right = right_tree
    return result_node


def count_leaves(tree):
    if tree.is_leaf:
        return 1
    return count_leaves(tree.left) + count_leaves(tree.right)


def predict_single_data(tree, x, annotate=False):
    # if the node is the leaf, return
    if tree.is_leaf:
        if annotate:
            print("leaf node, predicting %s" % tree.prediction)
        return tree.prediction
    else:
        # check the feature of current node
        split_feature_value = x[tree.split_feature]

        if annotate:
            print("Split on %s = %s" % (tree.split_feature, split_feature_value))
        if split_feature_value == 0:
            # if the feature of the data is 0, predict on the left split
            return predict_single_data(tree.left, x, annotate)
        else:
            # if the feature of the data is 1, predict on the right split
            return predict_single_data(tree.right, x, annotate)


def evaluate_accuracy(tree, data):
    # using the function of prediction on each row of data
    prediction = data.apply(lambda row: predict_single_data(tree, row), axis=1)
    # return the accuracy
    accuracy = (prediction == data['class']).sum() * 1.0 / len(data)
    return accuracy


def main():
    np.random.seed(9)

    data_folder = "../input"
    data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"))

    "process the input data"
    target = 'class'
    data[target] = data.apply(lambda row: -1 if row[0] == 'e' else 1, axis=1)
    cols = data.columns.drop(target)
    data_set = dummies(data, columns=cols)

    "split the input data into training data and testing data"
    train_data, test_data = train_test_split(data_set, test_size=0.3)

    trainX, trainY = train_data[train_data.columns[1:]], pd.DataFrame(train_data[target])
    testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])

    #test
    example_targets = np.array([-1, -1, 1, 1, 1])
    example_data_weights = np.array([1., 2., .5, 1., 1.])
    node_weighted_mistakes(example_targets, example_data_weights)
    # (2.5, -1)

    # test
    features = data_set.columns.drop(target)
    example_data_weights = np.array(len(train_data) * [2])
    best_split_weighted(train_data, features, target, example_data_weights)
    # ('odor_n')

    # test
    example_data_weights = np.array([1.0 for i in range(len(train_data))])
    small_data_decision_tree = create_weighted_tree(train_data, example_data_weights, features, target, max_depth=2)
    count_leaves(small_data_decision_tree)


    evaluate_accuracy(small_data_decision_tree, test_data)

    m = MyAdaboost(10)
    m.fit(trainX, trainY)

    m.score(testX, testY)

if __name__ == '__main__':
    main()
