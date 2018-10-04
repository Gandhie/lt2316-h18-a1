# Module file for implementation of ID3 algorithm.

# You can add optional keyword parameters to anything, but the original
# interface must work with the original test file.
# You will of course remove the "pass".

import os, sys
import numpy as np
import dill
import pandas as pd
import sklearn.metrics as sk
# You can add any other imports you need.

class DecisionTree:
    def __init__(self, load_from=None):
        """Initialises the class. Loads the model from file if load_from is not "None".

        Args:
            load_from: an argument that if not set to "None" specifies a file object to load the model from.

        Returns:
            Does not return anything."""
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.tree = dill.load(load_from)

# The entropy(...), InformationGain(...) and train(...) functions were inspired
# by:
# - Title: What are Decision Trees?
# - Author: Schlagenhauf, T.
# - Date: Unknown.
# - Availability: https://www.python-course.eu/Decision_Trees.php

    def entropy(self, y):
        """Calculates total entropy.

        Args:
            y: A pd.Series of class labels.

        Returns:
            Returns the total entropy."""
        classes, count = np.unique(y, return_counts=True)
        entropy = 0
        for i in range(len(classes)):
            entropy += -(count[i]/np.sum(count) * np.log2(count[i]/np.sum(count)))
        return entropy

    def InformationGain(self, X, y, split_attr):
        """Calculates the information gain of a dataset for a given attribute.

        Args:
            X: the dataset, without classes.
            y: A pd.Series of class labels.
            split_attr: the attribute for which to calculate information gain.

        Returns:
            The information gain for the given attribute."""
        total_entropy = DecisionTree.entropy(self, y)
        values, count = np.unique(X[split_attr], return_counts=True)
        weighted_entropy = np.sum([(count[i]/np.sum(count)) * DecisionTree.entropy(self, y.where(X[split_attr]==values[i]).dropna()) for i in range(len(values))])
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def train(self, X, y, attrs, prune=False):
        """Trains the class and builds the ID3 Decision Tree model.

        Args:
            X: the dataset, without classes.
            y: A pd.Series of class labels.
            attrs: a list of attributes.
            prune: argument to toggle pruning.

        Returns:
            Returns the ID3 tree."""
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.
        if len(np.unique(y)) <= 1:
            return np.unique(y)[0]
        elif len(attrs) == 0:
            return y.mode()
        else:
            infogains = [DecisionTree.InformationGain(self, X, y, attr) for attr in attrs]
            best_attr = attrs[np.argmax(infogains)]
            tree = {best_attr: {}}

            attrs = [i for i in attrs if i != best_attr]

            for value in np.unique(X[best_attr]):
                subdata_X = X.where(X[best_attr] == value).dropna()
                subdata_y = y.where(X[best_attr] == value).dropna()
                subtree = DecisionTree.train(self, subdata_X, subdata_y, attrs)
                tree[best_attr][value] = subtree

        self.tree = tree
        return self.tree

    def predict(self, instance, tree, y):
        """Predicts the label for a given instance based on the ID3 tree.

        Args:
            instance: an instance with attributes and values, to be classified.
            tree: the ID3 tree.
            y: a pd.Series of class labels.

        Returns:
            Returns the predicted class label."""
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.
        if isinstance(self.tree, dict):
            for attr in instance.keys():
                if attr in tree.keys():
                    try:
                        prediction = tree[attr][instance[attr]]
                    except:
                        return y.mode()
                    if isinstance(prediction, dict):
                        prediction = DecisionTree.predict(self, instance, prediction, y)
                        return prediction
                    else:
                        return prediction
        else:
            raise ValueError('ID3 untrained')

    def test(self, X, y, display=False):
        """Gets predicted labels for a set of instances and compares to gold standard to calculate accuracy, precision, recall, F1-score, and getting a confusion matrix.

        Args:
            X: the dataset, without classes.
            y: a pd.Series of class labels.
            display: argument to toggle whether to print the resulting scores.

        Returns:
            Returns the scores in a dict."""
        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.
        instances = X.to_dict(orient = 'records')
        prediction_list = []
        default = y.mode()
        for instance in instances:
            prediction_list.append(DecisionTree.predict(self, instance, self.tree, default))
        predictions = pd.Series(prediction_list)

        result = {'precision': sk.precision_score(y, predictions, average="micro"),
                  'recall': sk.recall_score(y, predictions, average="micro"),
                  'accuracy': sk.accuracy_score(y, predictions),
                  'F1': sk.f1_score(y, predictions, average="micro"),
                  'confusion-matrix': sk.confusion_matrix(y, predictions)}
        if display:
            print(result)
        return result

    def __str__(self):
        """Checks if the class is trained and returns the trained model in a readable format if it is, and an error message if it is not.

        Args:
            Does not take any arguments.

        Returns:
            Returns the trained class, or an error message."""
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        if isinstance(self.tree, dict):
            return str(self.tree)
        else:
            return "ID3 untrained"

    def save(self, output):
        """Saves the trained model to a file.

        Args:
            output: a file object to which to save the model.

        Returns:
            Does not return anything."""
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        dill.dump(self.tree, output)
