import numpy as np


class DecisionPath:
    RIGHT_SIGN = '<='
    LEFT_SIGN = '>'

    def __init__(self):
        self.decision_node_list = list()
        self.threshold_sign_list = list()

    @staticmethod
    def generate(classifier=None, decision_node_list=None, x=None):
        if (classifier is None) or (decision_node_list is None) or (x is None):
            raise Exception('Invalid parameters')

        decision_path = DecisionPath()
        decision_path.decision_node_list = DecisionPath.get_decision_node_list(classifier, decision_node_list, x)
        decision_path.threshold_sign_list = DecisionPath.get_threshold_sign_list(decision_path.decision_node_list, x)

        return decision_path

    @staticmethod
    def get_decision_node_list(classifier, decision_node_list, x):
        decision_path = list()

        node_indicator = classifier.decision_path(x.reshape((1, -1)))
        node_ids = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[0 + 1]]

        for node_id in node_ids:
            decision_path.append(decision_node_list[node_id])

        return decision_path

    @staticmethod
    def get_threshold_sign_list(decision_path, x):
        threshold_sign_list = list()

        for decision_node in decision_path:
            if x[decision_node.feature_index] <= decision_node.threshold:
                threshold_sign = DecisionPath.RIGHT_SIGN

            else:
                threshold_sign = DecisionPath.LEFT_SIGN

            threshold_sign_list.append(threshold_sign)

        return threshold_sign_list

    def get_rule_to_list(self, variable_name_list=None):
        rule_list = list()

        for index in range(0, len(self.decision_node_list) - 1):
            decision_node = self.decision_node_list[index]
            threshold_sign = self.threshold_sign_list[index]

            variable_name = decision_node.node_id
            if variable_name_list is not None:
                variable_name = variable_name_list[decision_node.node_id]

            rule_list.append((variable_name, threshold_sign, decision_node.threshold))

            if index == (len(self.decision_node_list) - 2):
                break

        return rule_list

    def get_str_decision_path(self, variable_name_list=None, class_name_list=None):
        decision_rules_str = 'if '

        for index in range(0, len(self.decision_node_list) - 1):
            decision_node = self.decision_node_list[index]
            threshold_sign = self.threshold_sign_list[index]

            variable_name = decision_node.node_id
            if variable_name_list is not None:
                variable_name = variable_name_list[decision_node.node_id]

            decision_rules_str += '({0} {1} {2})'.format(variable_name, threshold_sign, decision_node.threshold)

            if index == (len(self.decision_node_list) - 2):
                break

            decision_rules_str += ' and '

        decision_rules_str += ' then '

        leaf_node = self.decision_node_list[-1]
        class_index = np.argmax(leaf_node.number_of_predict_per_class)

        class_name = class_index
        if class_name_list is not None:
            class_name = class_name_list[class_index]

        probability = np.round(100.0 * leaf_node.number_of_predict_per_class[class_index] / np.sum(leaf_node.number_of_predict_per_class), 2)
        number_of_samples = leaf_node.number_of_train_sample

        decision_rules_str += 'class: {0} (probability: {1}%) | based on {2} samples'.format(class_name, probability,number_of_samples)

        return decision_rules_str
