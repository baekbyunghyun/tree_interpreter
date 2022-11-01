import re
import numpy as np

from lifelog_analysis.service.lifestyle_guide.utils.utils import *
from lifelog_analysis.service.lifestyle_guide.interpret.tree.boosting_decision_node import BoostingDecisionNode


class BoostingTree:
    def __init__(self):
        self.class_index = None
        self.boosting_round = None
        self.type_of_classifier = None

        self.boosting_decision_node_list = list()

    @staticmethod
    def generate(index=None, number_of_classes=None, tree_dump=None, type_of_classifier=None):
        if (index is None) or (number_of_classes is None) or (tree_dump is None) or (type_of_classifier is None):
            raise Exception("Invalid parameters for generate boosting tree.")

        boosting_tree = BoostingTree()

        if number_of_classes > 2:
            boosting_tree.class_index = index % number_of_classes
            boosting_tree.boosting_round = index // number_of_classes

        else:
            boosting_tree.class_index = 1
            boosting_tree.boosting_round = index

        boosting_tree.type_of_classifier = type_of_classifier

        boosting_tree._set_boosting_decision_node_list(tree_dump, type_of_classifier)

        return boosting_tree

    def _set_boosting_decision_node_list(self, tree_dump, type_of_classifier):
        if type_of_classifier == "xgboost":
            self._set_decision_node_list_for_xgboost(tree_dump)

        elif type_of_classifier == "lightgbm":
            self._set_decision_node_list_for_lightgbm(tree_dump["tree_structure"], node_id=0)

        else:
            raise Exception("Invalid classifier type.")

    def _set_decision_node_list_for_xgboost(self, tree_dump):
        regex = re.compile(BoostingDecisionNode.DECISION_NODE_REGULAR_EXPRESSION)
        values = regex.findall(tree_dump)

        self.boosting_decision_node_list = [BoostingDecisionNode() for _ in range(0, len(values))]

        for value in values:
            node_id = get_str_to_integer(value[0])

            boosting_decision_node = self.boosting_decision_node_list[node_id]

            boosting_decision_node.node_id = node_id
            boosting_decision_node.feature_index = get_str_to_integer(value[2])
            boosting_decision_node.threshold = get_str_to_float(value[3])
            boosting_decision_node.margin = get_str_to_float(value[5])
            boosting_decision_node.gain = get_str_to_float(value[10])
            boosting_decision_node.cover = get_str_to_float(value[11])
            boosting_decision_node.yes_node_id = get_str_to_integer(value[7])
            boosting_decision_node.no_node_id = get_str_to_integer(value[8])
            boosting_decision_node.missing_node_id = get_str_to_integer(value[9])

    def _set_decision_node_list_for_lightgbm(self, tree_dump, node_id):
        boosting_decision_node = BoostingDecisionNode()

        boosting_decision_node.node_id = node_id
        boosting_decision_node.feature_index = get_value_in_dict(key="split_feature", value_dict=tree_dump)
        boosting_decision_node.threshold = get_value_in_dict(key="threshold", value_dict=tree_dump)
        boosting_decision_node.gain = get_value_in_dict(key="split_gain", value_dict=tree_dump)
        boosting_decision_node.number_of_train_data = get_value_in_dict_using_multiple_keys(key_list=["internal_count", "leaf_count"], value_dict=tree_dump)
        boosting_decision_node.margin = get_value_in_dict_using_multiple_keys(key_list=["internal_value", "leaf_value"], value_dict=tree_dump)
        boosting_decision_node.weight = get_value_in_dict_using_multiple_keys(key_list=["internal_weight", "leaf_weight"], value_dict=tree_dump)

        self.boosting_decision_node_list.append(boosting_decision_node)

        if boosting_decision_node.feature_index is not None:
            next_node_id = self._set_decision_node_list_for_lightgbm(tree_dump["left_child"], node_id + 1)
            no_node_id = next_node_id + 1

            boosting_decision_node.yes_node_id = node_id + 1
            boosting_decision_node.missing_node_id = node_id + 1
            boosting_decision_node.no_node_id = no_node_id

            no_node_id = self._set_decision_node_list_for_lightgbm(tree_dump["right_child"], no_node_id)

            return no_node_id

        return node_id

    def predict(self, test_data_list):
        predict_function = self._get_predict_function()

        if len(test_data_list.shape) == 1:
            test_data_list = test_data_list.reshape(1, -1)

        output_margin_list = list()
        for x in test_data_list:
            leaf_node = predict_function(x)
            output_margin = leaf_node.margin

            output_margin_list.append(output_margin)

        return output_margin_list

    def _get_predict_function(self):
        if self.type_of_classifier == "xgboost":
            predict_function = self._get_leaf_node_for_prediction_for_xgboost

        elif self.type_of_classifier == "lightgbm":
            predict_function = self._get_leaf_node_for_prediction_for_lightgbm

        else:
            raise Exception("Invalid classifier type.")

        return predict_function

    def _get_leaf_node_for_prediction_for_xgboost(self, x):
        boosting_node = self.boosting_decision_node_list[0]

        while True:
            if boosting_node.margin is not None:
                break

            if x[boosting_node.feature_index] < boosting_node.threshold:
                child_node_index = boosting_node.yes_node_id

            else:
                child_node_index = boosting_node.no_node_id

            boosting_node = self.boosting_decision_node_list[child_node_index]

        return boosting_node

    def _get_leaf_node_for_prediction_for_lightgbm(self, x):
        boosting_node = self.boosting_decision_node_list[0]

        while True:
            if boosting_node.feature_index is None:
                break

            if x[boosting_node.feature_index] <= boosting_node.threshold:
                child_node_index = boosting_node.yes_node_id

            else:
                child_node_index = boosting_node.no_node_id

            boosting_node = self.boosting_decision_node_list[child_node_index]

        return boosting_node

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = '------------ {0}th boosting round Tree ------------'.format(self.boosting_round)
        msg += "\n Class document:                 {0}".format(self.class_index)
        msg += "\n Boosting round:                 {0}".format(self.boosting_round)
        msg += "\n Number of nodes:                {0}".format(len(self.boosting_decision_node_list))
        msg += "\n\n"

        return msg
