import copy

from interpret.tree.boosting_decision_path import BoostingDecisionPath
from branch.finder.branch_boosting_decision_node_finder import BranchBoostingDecisionNodeFinder


class BoostingTreeInterpreter:
    def __init__(self, boosting_tree=None, variable_name_list=None, class_name_list=None, type_of_classifier=None):
        if (boosting_tree is None) or (variable_name_list is None) or (class_name_list is None) or (type_of_classifier is None):
            raise Exception("Invalid parameters for create boosting tree interpreter")

        self.boosting_tree = boosting_tree
        self.variable_name_list = variable_name_list
        self.class_name_list = class_name_list
        self.type_of_classifier = type_of_classifier

        self.decision_path_list = self._set_deicision_path_list(type_of_classifier)

        self.branch_boosting_decision_node_finder = BranchBoostingDecisionNodeFinder(decision_path_list=self.decision_path_list,
                                                                                     used_variable_name_list=self.variable_name_list)

    def _set_deicision_path_list(self, type_of_classifier):
        if type_of_classifier == "xgboost":
            decision_path_list = self._get_decision_path_list_of_tree_for_xgboost()

        elif type_of_classifier == "lightgbm":
            decision_path_list = self._get_decision_path_list_of_tree_for_lightgbm()

        else:
            raise Exception("Invalid classifier type. you must input either 'xgboost' or 'lightgbm'.")

        return decision_path_list

    def _get_decision_path_list_of_tree_for_xgboost(self):
        boosting_decision_path_list = list()
        boosting_decision_path = BoostingDecisionPath()

        def find_boosting_decision_rule(node_id, boosting_decision_path, boosting_decision_path_list):
            node = self.boosting_tree.boosting_decision_node_list[node_id]

            if node.margin is None:
                left_boosting_decision_path = copy.deepcopy(boosting_decision_path)
                left_boosting_decision_path.boosting_decision_node_list.append(node)
                left_boosting_decision_path.threshold_sign_list.append(BoostingDecisionPath.LEFT_SIGN_OF_XGBOOST)

                find_boosting_decision_rule(node.yes_node_id, left_boosting_decision_path, boosting_decision_path_list)

                right_boosting_decision_path = copy.deepcopy(boosting_decision_path)
                right_boosting_decision_path.boosting_decision_node_list.append(node)
                right_boosting_decision_path.threshold_sign_list.append(BoostingDecisionPath.RIGHT_SIGN_OF_XGBOOST)

                find_boosting_decision_rule(node.no_node_id, right_boosting_decision_path, boosting_decision_path_list)

            else:
                boosting_decision_path.boosting_decision_node_list.append(node)
                boosting_decision_path_list.append(boosting_decision_path)

        find_boosting_decision_rule(0, boosting_decision_path, boosting_decision_path_list)

        return boosting_decision_path_list

    def _get_decision_path_list_of_tree_for_lightgbm(self):
        boosting_decision_path_list = list()
        boosting_decision_path = BoostingDecisionPath()

        def find_boosting_decision_rule(node_id, boosting_decision_path, boosting_decision_path_list):
            node = self.boosting_tree.boosting_decision_node_list[node_id]

            if node.feature_index is not None:
                left_boosting_decision_path = copy.deepcopy(boosting_decision_path)
                left_boosting_decision_path.boosting_decision_node_list.append(node)
                left_boosting_decision_path.threshold_sign_list.append(BoostingDecisionPath.RIGHT_SIGN_OF_LIGHTGBM)

                find_boosting_decision_rule(node.yes_node_id, left_boosting_decision_path, boosting_decision_path_list)

                right_boosting_decision_path = copy.deepcopy(boosting_decision_path)
                right_boosting_decision_path.boosting_decision_node_list.append(node)
                right_boosting_decision_path.threshold_sign_list.append(BoostingDecisionPath.LEFT_SIGN_OF_LIGHTGBM)

                find_boosting_decision_rule(node.no_node_id, right_boosting_decision_path, boosting_decision_path_list)

            else:
                boosting_decision_path.boosting_decision_node_list.append(node)
                boosting_decision_path_list.append(boosting_decision_path)

        find_boosting_decision_rule(0, boosting_decision_path, boosting_decision_path_list)

        return boosting_decision_path_list


    def find_branch_boosting_decision_node_list(self, x=None, y=None, optimal_class_find_func=None):
        if (x is None) or (y is None) or (optimal_class_find_func is None):
            raise Exception('Invalid parameter for find branch interpret node')

        class_index = self.boosting_tree.class_index
        if class_index != optimal_class_find_func(y):
            return None

        decision_path_of_prediction = self.get_decision_path_for_prediction(x=x)
        branch_boosting_decision_node_list = self.branch_boosting_decision_node_finder.find(decision_path_of_prediction)

        return branch_boosting_decision_node_list

    def get_decision_path_for_prediction(self, x=None):
        if x is None:
            raise Exception('Invalid parameter')

        boosting_decision_path = BoostingDecisionPath.generate(boosting_tree=self.boosting_tree, x=x, type_of_classifier=self.type_of_classifier)

        return boosting_decision_path