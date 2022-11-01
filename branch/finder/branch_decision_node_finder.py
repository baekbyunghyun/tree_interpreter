import numpy as np


class BranchDecisionNode:
    def __init__(self, similarity=None, node=None, decision_path_of_similarity=None,
                 decision_path_of_prediction=None, variable_name=None):
        self.similarity = similarity
        self.node = node
        self.decision_path_of_similarity = decision_path_of_similarity
        self.decision_path_of_prediction = decision_path_of_prediction
        self.variable_name = variable_name

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = 'BranchDecisionNode INFO'
        msg += '\n> Similarity:                       {0}'.format(self.similarity)
        msg += '\n> Node id:                          {0}'.format(self.node.node_id)
        msg += '\n> Node feature name:                {0}'.format(self.variable_name)
        msg += '\n> Node feature document:               {0}'.format(self.node.feature_index)
        msg += '\n> Node threshold:                   {0}'.format(self.node.threshold)
        msg += '\n> Node gini:                        {0}'.format(self.node.gini)
        msg += '\n> Node number of train samples:     {0}'.format(self.node.number_of_train_sample)
        msg += '\n> Node number of predict per class: {0}'.format(self.node.number_of_predict_per_class)

        return msg


class BranchDecisionNodeFinder:
    def __init__(self, decision_path_list=None, decision_paths_dict_per_class=None, used_variable_name_list=None):
        if (decision_path_list is None) or (decision_paths_dict_per_class is None) or (used_variable_name_list is None):
            raise Exception('Invalid parameters of initialize')

        self.decision_path_list = decision_path_list
        self.decision_paths_dict_per_class = decision_paths_dict_per_class
        self.used_variable_name_list = used_variable_name_list

    def find(self, decision_path_of_prediction=None, optimal_class_find_func=None):
        if decision_path_of_prediction is None:
            raise Exception('Invalid decision path of prediction of parameter')

        if optimal_class_find_func is None:
            raise Exception('Invalid optimal class find function of parameter')

        branch_decision_node_list = self._initialize_branch_decision_node_list(decision_path_of_prediction, optimal_class_find_func)
        if len(branch_decision_node_list) <= 0:
            return None

        self._set_similarity_of_branch_decision_node(decision_path_of_prediction, branch_decision_node_list)
        self._find_branch_node_of_similar_decision_path(branch_decision_node_list)

        return branch_decision_node_list

    def _initialize_branch_decision_node_list(self, decision_path_of_prediction, optimal_class_find_func):
        leaf_node = decision_path_of_prediction.decision_node_list[-1]
        class_index_of_prediction = np.argmax(leaf_node.number_of_predict_per_class)

        optimal_class_index = optimal_class_find_func(class_index_of_prediction)
        decision_path_list_per_optimal_class = self.decision_paths_dict_per_class[optimal_class_index]

        branch_decision_node_list = list()
        for decision_path in decision_path_list_per_optimal_class:
            branch_decision_node = BranchDecisionNode()
            branch_decision_node.decision_path_of_prediction = decision_path_of_prediction
            branch_decision_node.decision_path_of_similarity = decision_path

            branch_decision_node_list.append(branch_decision_node)

        return branch_decision_node_list

    def _set_similarity_of_branch_decision_node(self, decision_path_of_prediction, branch_decision_node_list):
        for checked_node_index in range(0, len(decision_path_of_prediction.decision_node_list)):
            node_of_prediction = decision_path_of_prediction.decision_node_list[checked_node_index]
            rank_of_similarity = self._get_rank_of_similarity(branch_decision_node_list)

            for branch_decision_node in branch_decision_node_list:
                if branch_decision_node.similarity is not None:
                    continue

                decision_path_of_similarity = branch_decision_node.decision_path_of_similarity
                node_of_similarity = decision_path_of_similarity.decision_node_list[checked_node_index]

                if node_of_prediction.node_id != node_of_similarity.node_id:
                    branch_decision_node.similarity = rank_of_similarity

    def _get_rank_of_similarity(self, branch_decision_node_list):
        rank_of_similarity = 0

        for branch_decision_node in branch_decision_node_list:
            if branch_decision_node.similarity is None:
                rank_of_similarity += 1

        return rank_of_similarity

    def _find_branch_node_of_similar_decision_path(self, branch_decision_node_list):
        for branch_decision_node in branch_decision_node_list:
            decision_path_of_similarity = branch_decision_node.decision_path_of_similarity
            decision_path_of_prediction = branch_decision_node.decision_path_of_prediction

            branch_node = self._get_branch_node(decision_path_of_similarity, decision_path_of_prediction)

            branch_decision_node.node = branch_node
            branch_decision_node.variable_name = self.used_variable_name_list[branch_node.node_id]

    def _get_branch_node(self, decision_path_of_similarity, decision_path_of_prediction):
        branch_node = None

        for node_of_similarity, node_of_prediction in zip(decision_path_of_similarity.decision_node_list, decision_path_of_prediction.decision_node_list):
            if node_of_similarity.node_id != node_of_prediction.node_id:
                break

            branch_node = node_of_similarity

        return branch_node