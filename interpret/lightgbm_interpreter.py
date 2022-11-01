from interpret.boosting_tree_interpreter import BoostingTreeInterpreter


class LightGBMInterpreter:
    CLASSIFIER_NAME = "lightgbm"

    def __init__(self, classifier=None, variable_name_list=None, class_name_list=None):
        if (classifier is None) or (variable_name_list is None) or (class_name_list is None):
            raise Exception("Invalid parameter for initialize lightgbm interpreter.")

        self.classifier = classifier
        self.variable_name_list = variable_name_list
        self.class_name_list = class_name_list

        self.boosting_tree_interpreter_list = self._get_boosting_tree_interpreter_list()

    def _get_boosting_tree_interpreter_list(self):
        boosting_tree_interpreter_list = list()

        boosting_tree_list = self.classifier.boosting_tree_list
        for boosting_tree in boosting_tree_list:
            boosting_tree_interpreter = BoostingTreeInterpreter(boosting_tree=boosting_tree,
                                                                variable_name_list=self.variable_name_list,
                                                                class_name_list=self.class_name_list,
                                                                type_of_classifier=self.CLASSIFIER_NAME)

            boosting_tree_interpreter_list.append(boosting_tree_interpreter)

        return boosting_tree_interpreter_list

    def find_branch_decision_node_per_estimator(self, x=None, y=None, optimal_class_find_func=None):
        if (x is None) or (y is None) or (optimal_class_find_func is None):
            raise Exception('Invalid parameter for find branch interpret node')

        branch_boosting_decision_node_per_estimator_dict = dict()

        for boosting_tree_interpreter in self.boosting_tree_interpreter_list:
            branch_boosting_decision_node_list = boosting_tree_interpreter.find_branch_boosting_decision_node_list(x=x, y=y,
                                                                                                                   optimal_class_find_func=optimal_class_find_func)

            if branch_boosting_decision_node_list is None:
                continue

            branch_boosting_decision_node_per_estimator_dict[boosting_tree_interpreter] = branch_boosting_decision_node_list

        return branch_boosting_decision_node_per_estimator_dict

