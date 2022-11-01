import numpy as np


class BranchBoostingDecisionNode:
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
        msg += '\n> Node gain:                        {0}'.format(self.node.gain)
        msg += '\n> Node cover:                       {0}'.format(self.node.cover)
        msg += '\n> Node margin:                      {0}'.format(self.node.margin)

        return msg


class BranchBoostingDecisionNodeFinder:
    def __init__(self, decision_path_list=None, used_variable_name_list=None):
        if (decision_path_list is None) or (used_variable_name_list is None):
            raise Exception('Invalid parameters of initialize')

        self.decision_path_list = decision_path_list
        self.used_variable_name_list = used_variable_name_list

    def find(self, decision_path_of_prediction=None):
        if decision_path_of_prediction is None:
            raise Exception('Invalid decision path of prediction of parameter')

        branch_boosting_decision_node_list = self._initialize_branch_boosting_decision_node_list(decision_path_of_prediction)
        if len(branch_boosting_decision_node_list) <= 0:
            return None

        self._set_similarity_of_branch_boosting_decision_node(decision_path_of_prediction, branch_boosting_decision_node_list)
        self._find_branch_node_of_similar_decision_path(branch_boosting_decision_node_list)

        return branch_boosting_decision_node_list

    def _initialize_branch_boosting_decision_node_list(self, decision_path_of_prediction):
        leaf_node = decision_path_of_prediction.boosting_decision_node_list[-1]
        margin_of_prediction = leaf_node.margin

        branch_boosting_decision_node_list = list()
        for decision_path in self.decision_path_list:
            leaf_node = decision_path.boosting_decision_node_list[-1]
            if (leaf_node.margin <= 0) or (leaf_node.margin <= margin_of_prediction):
                continue

            branch_boosting_decision_node = BranchBoostingDecisionNode()
            branch_boosting_decision_node.decision_path_of_prediction = decision_path_of_prediction
            branch_boosting_decision_node.decision_path_of_similarity = decision_path

            branch_boosting_decision_node_list.append(branch_boosting_decision_node)

        return branch_boosting_decision_node_list

    def _set_similarity_of_branch_boosting_decision_node(self, decision_path_of_prediction, branch_boosting_decision_node_list):
        for checked_node_index in range(0, len(decision_path_of_prediction.boosting_decision_node_list)):
            node_of_prediction = decision_path_of_prediction.boosting_decision_node_list[checked_node_index]
            rank_of_similarity = self._get_rank_of_similarity(branch_boosting_decision_node_list)

            for branch_boosting_decision_node in branch_boosting_decision_node_list:
                if branch_boosting_decision_node.similarity is not None:
                    continue

                decision_path_of_similarity = branch_boosting_decision_node.decision_path_of_similarity
                node_of_similarity = decision_path_of_similarity.boosting_decision_node_list[checked_node_index]

                if node_of_prediction.node_id != node_of_similarity.node_id:
                    branch_boosting_decision_node.similarity = rank_of_similarity

    def _get_rank_of_similarity(self, branch_boosting_decision_node_list):
        rank_of_similarity = 0

        for branch_boosting_decision_node in branch_boosting_decision_node_list:
            if branch_boosting_decision_node.similarity is None:
                rank_of_similarity += 1

        return rank_of_similarity

    def _find_branch_node_of_similar_decision_path(self, branch_boosting_decision_node_list):
        for branch_boosting_decision_node in branch_boosting_decision_node_list:
            decision_path_of_similarity = branch_boosting_decision_node.decision_path_of_similarity
            decision_path_of_prediction = branch_boosting_decision_node.decision_path_of_prediction

            branch_node = self._get_branch_node(decision_path_of_similarity, decision_path_of_prediction)

            branch_boosting_decision_node.node = branch_node
            branch_boosting_decision_node.variable_name = self.used_variable_name_list[branch_node.feature_index]

    def _get_branch_node(self, decision_path_of_similarity, decision_path_of_prediction):
        branch_node = None

        for node_of_similarity, node_of_prediction in zip(decision_path_of_similarity.boosting_decision_node_list, decision_path_of_prediction.boosting_decision_node_list):
            if node_of_similarity.node_id != node_of_prediction.node_id:
                break

            branch_node = node_of_similarity

        return branch_node
