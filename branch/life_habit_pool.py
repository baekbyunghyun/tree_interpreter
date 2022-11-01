from branch.life_habit import LifeHabit


class LifeHabitPool:
    def __init__(self):
        self.life_habit_dict = dict()

    @staticmethod
    def generate_life_habit_pool(interpreter=None, x=None, y=None, optimal_class_find_func=None):
        if interpreter is None:
            raise Exception('Invalid parameter of interpreter')

        if x is None:
            raise Exception('Invalid parameter of dataset')

        if optimal_class_find_func is None:
            raise Exception('Invalid parameter of optimal class find function')

        if y is None:
            y = interpreter.classifier.predict(x.reshape(1, -1))

        life_habit_pool = LifeHabitPool()

        branch_decision_nodes_per_estimator_dict = interpreter.find_branch_decision_node_per_estimator(x=x, y=y, optimal_class_find_func=optimal_class_find_func)
        for _, branch_decision_node_list in branch_decision_nodes_per_estimator_dict.items():
            if (branch_decision_node_list is None) or (len(branch_decision_node_list) <= 0):
                continue

            life_habit_pool.parse_branch_decision_nodes(x, branch_decision_node_list)

        return life_habit_pool

    def parse_branch_decision_nodes(self, x, branch_decision_node_list):
        for branch_decision_node in branch_decision_node_list:
            variable_name = branch_decision_node.variable_name
            value = x[branch_decision_node.node.feature_index]
            similarity = branch_decision_node.similarity
            threshold = branch_decision_node.node.threshold

            if variable_name not in self.life_habit_dict:
                self.life_habit_dict[variable_name] = LifeHabit.generate(variable_name, value, similarity, threshold)

                continue

            life_habit = self.life_habit_dict[variable_name]
            life_habit.similarity_list.append(similarity)
            life_habit.threshold_list.append(threshold)

    def merge(self, life_habit_pool=None):
        if life_habit_pool is None:
            return

        for variable_name, life_habit in life_habit_pool.life_habit_dict.items():
            if variable_name not in self.life_habit_dict:
                self.life_habit_dict[variable_name] = life_habit

                continue

            self.life_habit_dict[variable_name].value_list += life_habit.value_list
            self.life_habit_dict[variable_name].similarity_list += life_habit.similarity_list
            self.life_habit_dict[variable_name].threshold_list += life_habit.threshold_list
