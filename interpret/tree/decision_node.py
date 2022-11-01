import numpy as np


class DecisionNode:
    def __init__(self):
        self.node_id = None
        self.feature_index = None
        self.threshold = None
        self.gini = None
        self.number_of_train_sample = None
        self.number_of_predict_per_class = None

    @staticmethod
    def get_decision_nodes(tree=None):
        if tree is None:
            raise Exception('Decision tree with training is none')

        node_id = 0
        feature_indexes = tree.feature
        thresholds = tree.threshold
        ginis = tree.impurity
        number_of_train_sample_list = tree.n_node_samples
        number_of_predict_per_class_list = tree.value

        decision_nodes = list()

        for feature_index, threshold, gini, number_of_train_sample, number_of_predict_per_class \
                in zip(feature_indexes, thresholds, ginis, number_of_train_sample_list, number_of_predict_per_class_list):
            decision_node = DecisionNode()
            decision_node.node_id = node_id
            decision_node.feature_index = feature_index
            decision_node.threshold = round(threshold, 3)
            decision_node.gini = round(gini, 3)
            decision_node.number_of_train_sample = number_of_train_sample
            decision_node.number_of_predict_per_class = np.asarray(number_of_predict_per_class).reshape(-1)

            decision_nodes.append(decision_node)

            node_id += 1

        return decision_nodes

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = '------------ Information of Decision {0}th node ------------'.format(self.node_id)
        msg += "\n node id:                     {0}".format(self.node_id)
        msg += "\n feature document:               {0}".format(self.feature_index)
        msg += "\n threshold:                   {0}".format(self.threshold)
        msg += "\n gini:                        {0}".format(self.gini)
        msg += "\n number of train sample:      {0}".format(self.number_of_train_sample)
        msg += "\n number_of_predict_per_class: {0}".format(self.number_of_predict_per_class)
        msg += "\n\n"

        return msg
