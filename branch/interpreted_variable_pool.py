from branch.interpreted_variable import InterpretedVariable


class InterpretedVariablePool:
    def __init__(self):
        self.interpreted_variable_dict = dict()

    @staticmethod
    def generate_interpreted_variable_pool(interpreter=None, x=None, y=None, optimal_class_find_func=None):
        if interpreter is None:
            raise Exception('Invalid parameter of interpreter')

        if x is None:
            raise Exception('Invalid parameter of dataset')

        if optimal_class_find_func is None:
            raise Exception('Invalid parameter of optimal class find function')

        if y is None:
            y = interpreter.classifier.predict(x.reshape(1, -1))

        interpreted_variable_pool = InterpretedVariablePool()

        branch_decision_nodes_per_estimator_dict = interpreter.find_branch_decision_node_per_estimator(x=x, y=y, optimal_class_find_func=optimal_class_find_func)
        for _, branch_decision_node_list in branch_decision_nodes_per_estimator_dict.items():
            if (branch_decision_node_list is None) or (len(branch_decision_node_list) <= 0):
                continue

            interpreted_variable_pool.parse_branch_decision_nodes(x, branch_decision_node_list)

        return interpreted_variable_pool

    def parse_branch_decision_nodes(self, x, branch_decision_node_list):
        for branch_decision_node in branch_decision_node_list:
            variable_name = branch_decision_node.variable_name
            value = x[branch_decision_node.node.feature_index]
            similarity = branch_decision_node.similarity
            threshold = branch_decision_node.node.threshold

            if variable_name not in self.interpreted_variable_dict:
                self.interpreted_variable_dict[variable_name] = InterpretedVariable.generate(variable_name, value, similarity, threshold)

                continue

            interpreted_variable = self.interpreted_variable_dict[variable_name]
            interpreted_variable.similarity_list.append(similarity)
            interpreted_variable.threshold_list.append(threshold)

    def merge(self, interpreted_variable_pool=None):
        if interpreted_variable_pool is None:
            return

        for variable_name, interpreted_variable in interpreted_variable_pool.interpreted_variable_dict.items():
            if variable_name not in self.interpreted_variable_dict:
                self.interpreted_variable_dict[variable_name] = interpreted_variable

                continue

            self.interpreted_variable_dict[variable_name].value_list += interpreted_variable.value_list
            self.interpreted_variable_dict[variable_name].similarity_list += interpreted_variable.similarity_list
            self.interpreted_variable_dict[variable_name].threshold_list += interpreted_variable.threshold_list
