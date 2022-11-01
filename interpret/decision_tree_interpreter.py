import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.tree import _tree

from interpret.tree.decision_node import DecisionNode
from interpret.tree.decision_path import DecisionPath
from branch.finder.branch_decision_node_finder import BranchDecisionNodeFinder


class DecisionTreeInterpreter:
    def __init__(self, classifier=None, variable_name_list=None, class_name_list=None):
        if (classifier is None) or (variable_name_list is None) or (class_name_list is None):
            raise Exception('Invalid interpret tree interpreter parameters')

        self.decision_tree_classifier = classifier
        self.variable_name_list = variable_name_list
        self.class_name_list = class_name_list

        self.decision_node_list = DecisionNode.get_decision_nodes(self.decision_tree_classifier.tree_)
        self.used_variable_name_list = self._get_used_variable_name_list()
        self.decision_path_list = self._get_decision_path_list_of_tree()
        self.decision_paths_dict_per_class = self._get_decision_paths_dict_per_class()

        self.branch_node_finder = BranchDecisionNodeFinder(decision_path_list=self.decision_path_list,
                                                           decision_paths_dict_per_class=self.decision_paths_dict_per_class,
                                                           used_variable_name_list=self.used_variable_name_list)

    def _get_used_variable_name_list(self):
        tree = self.decision_tree_classifier.tree_

        used_variable_name_list = list()
        for feature_index in tree.feature:
            if feature_index != _tree.TREE_UNDEFINED:
                used_variable_name_list.append(self.variable_name_list[feature_index])

            else:
                used_variable_name_list.append("undefined")

        return used_variable_name_list

    def _get_decision_path_list_of_tree(self):
        decision_tree = self.decision_tree_classifier.tree_

        decision_path_list = list()
        decision_path = DecisionPath()

        def find_decision_rule(node_id, decision_path, decision_path_list):
            if decision_tree.feature[node_id] != _tree.TREE_UNDEFINED:
                left_decision_path = copy.deepcopy(decision_path)
                left_decision_path.decision_node_list.append(self.decision_node_list[node_id])
                left_decision_path.threshold_sign_list.append(DecisionPath.RIGHT_SIGN)

                find_decision_rule(decision_tree.children_left[node_id], left_decision_path, decision_path_list)

                right_decision_path = copy.deepcopy(decision_path)
                right_decision_path.decision_node_list.append(self.decision_node_list[node_id])
                right_decision_path.threshold_sign_list.append(DecisionPath.LEFT_SIGN)

                find_decision_rule(decision_tree.children_right[node_id], right_decision_path, decision_path_list)

            else:
                decision_path.decision_node_list.append(self.decision_node_list[node_id])
                decision_path_list.append(decision_path)

        find_decision_rule(0, decision_path, decision_path_list)

        number_of_samples_list = list()
        for decision_path in decision_path_list:
            leaf_node = decision_path.decision_node_list[-1]
            number_of_samples_list.append(leaf_node.number_of_train_sample)

        sort_index_list = list(np.argsort(number_of_samples_list))

        return [decision_path_list[index] for index in reversed(sort_index_list)]

    def _get_decision_paths_dict_per_class(self):
        decision_paths_dict_per_class = dict()

        for decision_path in self.decision_path_list:
            leaf_node = decision_path.decision_node_list[-1]
            class_index = np.argmax(leaf_node.number_of_predict_per_class)

            if class_index not in decision_paths_dict_per_class:
                decision_paths_dict_per_class[class_index] = list()

            decision_paths_dict_per_class[class_index].append(decision_path)

        return decision_paths_dict_per_class

    def find_branch_decision_node_list(self, x=None, y=None, optimal_class_find_func=None):
        if (x is None) or (y is None) or (optimal_class_find_func is None):
            raise Exception('Invalid parameter for find branch interpret node')

        decision_path = self.get_decision_path_for_prediction(x=x)

        leaf_node = decision_path.decision_node_list[-1]
        class_index = np.argmax(leaf_node.number_of_predict_per_class)

        if y != class_index:
            return None

        branch_decision_node_list = self.branch_node_finder.find(decision_path, optimal_class_find_func)

        return branch_decision_node_list

    def get_decision_path_for_prediction(self, x=None):
        if x is None:
            raise Exception('Invalid parameter')

        decision_path = DecisionPath.generate(classifier=self.decision_tree_classifier,
                                              decision_node_list=self.decision_node_list,
                                              x=x)

        return decision_path

    def show_graphviz(self, output_file_path=None):
        if output_file_path is None:
            raise Exception('Invalid graphviz output path: {0}'.format(output_file_path))

        fig = plt.figure(figsize=(15, 15))
        plot_tree(self.decision_tree_classifier, filled=True)

        # plt.show()

        fig.savefig(output_file_path)
        plt.close(fig)
