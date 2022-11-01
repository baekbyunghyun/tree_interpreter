

class BoostingDecisionPath:
    RIGHT_SIGN_OF_XGBOOST = ">="
    LEFT_SIGN_OF_XGBOOST = "<"
    RIGHT_SIGN_OF_LIGHTGBM = "<="
    LEFT_SIGN_OF_LIGHTGBM = ">"

    def __init__(self):
        self.boosting_decision_node_list = list()
        self.threshold_sign_list = list()

    @staticmethod
    def generate(boosting_tree=None, x=None, type_of_classifier=None):
        if (boosting_tree is None) or (x is None) or (type_of_classifier is None):
            raise Exception('Invalid parameter for generate decision path')

        decision_path = BoostingDecisionPath()
        decision_path._set_info_in_decision_node(boosting_tree, x, type_of_classifier)

        return decision_path

    def _set_info_in_decision_node(self, boosting_tree, x, type_of_classifier):
        if type_of_classifier == "xgboost":
            self._set_info_in_decision_node_for_xgboost(boosting_tree, x)

        elif type_of_classifier == "lightgbm":
            self._set_info_in_decision_node_for_lightgbm(boosting_tree, x)

        else:
            raise Exception("Invalid classifier type. you must input either 'xgboost' or 'lightgbm'.")

    def _set_info_in_decision_node_for_xgboost(self, boosting_tree, x):
        self.boosting_decision_node_list = list()
        self.threshold_sign_list = list()

        boosting_node = boosting_tree.boosting_decision_node_list[0]

        while True:
            self.boosting_decision_node_list.append(boosting_node)

            if boosting_node.margin is not None:
                break

            if x[boosting_node.feature_index] < boosting_node.threshold:
                threshold_sign = BoostingDecisionPath.LEFT_SIGN_OF_XGBOOST
                child_node_index = boosting_node.yes_node_id

            else:
                threshold_sign = BoostingDecisionPath.RIGHT_SIGN_OF_XGBOOST
                child_node_index = boosting_node.no_node_id

            self.threshold_sign_list.append(threshold_sign)

            boosting_node = boosting_tree.boosting_decision_node_list[child_node_index]

    def _set_info_in_decision_node_for_lightgbm(self, boosting_tree, x):
        self.boosting_decision_node_list = list()
        self.threshold_sign_list = list()

        boosting_node = boosting_tree.boosting_decision_node_list[0]

        while True:
            self.boosting_decision_node_list.append(boosting_node)

            if boosting_node.feature_index is None:
                break

            if x[boosting_node.feature_index] <= boosting_node.threshold:
                threshold_sign = BoostingDecisionPath.RIGHT_SIGN_OF_LIGHTGBM
                child_node_index = boosting_node.yes_node_id

            else:
                threshold_sign = BoostingDecisionPath.LEFT_SIGN_OF_LIGHTGBM
                child_node_index = boosting_node.no_node_id

            self.threshold_sign_list.append(threshold_sign)

            boosting_node = boosting_tree.boosting_decision_node_list[child_node_index]

    def get_str_boosting_decision_path(self, variable_name_list=None):
        decision_rules_str = "if "

        for index in range(0, len(self.boosting_decision_node_list) - 1):
            boosting_decision_node = self.boosting_decision_node_list[index]
            threshold_sign = self.threshold_sign_list[index]

            variable_name = boosting_decision_node.node_id
            if variable_name_list is not None:
                variable_name = variable_name_list[boosting_decision_node.node_id]

            decision_rules_str += '({0} {1} {2})'.format(variable_name, threshold_sign, boosting_decision_node.threshold)

            if index == (len(self.boosting_decision_node_list) - 2):
                break

            decision_rules_str += ' and '

        decision_rules_str += ' then margin value: {0}'.format(self.boosting_decision_node_list[-1].margin)
