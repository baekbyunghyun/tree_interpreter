import os

from interpret.decision_tree_interpreter import DecisionTreeInterpreter


class RandomForestInterpreter:
    def __init__(self, classifier=None, variable_name_list=None, class_name_list=None):
        if (classifier is None) or (variable_name_list is None) or (class_name_list is None):
            raise Exception('Invalid parameter for initialize random forest interpreter')

        self.classifier = classifier
        self.variable_name_list = variable_name_list
        self.class_name_list = class_name_list

        self.decision_tree_interpreter_list = self._get_decision_tree_interpreter_list()

    def _get_decision_tree_interpreter_list(self):
        decision_tree_interpreter_list = list()

        for estimator in self.classifier.estimators_:
            decision_tree_interpreter = DecisionTreeInterpreter(classifier=estimator,
                                                                variable_name_list=self.variable_name_list,
                                                                class_name_list=self.class_name_list)

            decision_tree_interpreter_list.append(decision_tree_interpreter)

        return decision_tree_interpreter_list

    def find_branch_decision_node_per_estimator(self, x=None, y=None, optimal_class_find_func=None):
        if (x is None) or (y is None) or (optimal_class_find_func is None):
            raise Exception('Invalid parameter for find branch interpret node')

        branch_decision_nodes_per_estimator_dict = dict()

        for decision_tree_interpreter in self.decision_tree_interpreter_list:
            branch_decision_node_list = decision_tree_interpreter.find_branch_decision_node_list(x=x, y=y,
                                                                                                 optimal_class_find_func=optimal_class_find_func)
            if branch_decision_node_list is None:
                continue

            branch_decision_nodes_per_estimator_dict[decision_tree_interpreter] = branch_decision_node_list

        return branch_decision_nodes_per_estimator_dict

    def show_graphviz(self, output_dir_path=None):
        if output_dir_path is None:
            raise Exception('Invalid graphviz output directory path: {0}'.format(output_dir_path))

        index = 0
        for decision_tree_interpreter in self.decision_tree_interpreter_list:
            fig_path = os.path.join(output_dir_path, 'RandomForest_#{0}.png'.format(index))
            decision_tree_interpreter.show_graphviz(output_file_path=fig_path)

            index += 1
