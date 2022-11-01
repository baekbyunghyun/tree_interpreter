import numpy as np


class BranchDecisionNode:
    def __init__(self, branch_decision_node=None, decision_path_of_similarity=None, decision_path_of_prediction=None, boosting_round_of_tree=None, decision_weight=None, variable_name=None):
        if (branch_decision_node is None) or (decision_path_of_similarity is None) or (decision_path_of_prediction is None) or (decision_weight is None):
            raise Exception("Invalid paramters of initialize.")

        self.branch_decision_node = branch_decision_node
        self.decision_path_of_similarity = decision_path_of_similarity
        self.decision_path_of_prediction = decision_path_of_prediction
        self.boosting_round_of_tree = boosting_round_of_tree
        self.decision_weight = decision_weight
        self.variable_name = variable_name

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = 'BranchDecisionNode INFO'
        msg += '\n> Boosting round of tree:           {0}'.format(self.boosting_round_of_tree)
        msg += '\n> Decision weight:                  {0}'.format(self.decision_weight)
        msg += '\n> Node id:                          {0}'.format(self.branch_decision_node.node_id)
        msg += '\n> Node feature name:                {0}'.format(self.variable_name)
        msg += '\n> Node feature document:               {0}'.format(self.branch_decision_node.feature_index)
        msg += '\n> Node threshold:                   {0}'.format(self.branch_decision_node.threshold)
        msg += '\n> Node gini:                        {0}'.format(self.branch_decision_node.gain)
        msg += '\n> Node cover:                       {0}'.format(self.branch_decision_node.cover)
        msg += '\n> Node depth:                       {0}'.format(self.branch_decision_node.depth)

        return msg


class WeightLifePatternFinder:
    def __init__(self, boosting_tree_interpreter=None):
        self.boosting_tree_interpreter = boosting_tree_interpreter

        self.branch_decision_node_dict = self._set_branch_decision_node_dict()

    def find(self, leaf_node):
        if leaf_node.node_id not in self.branch_decision_node_dict.keys():
            return None

        branch_decision_node_list = self.branch_decision_node_dict[leaf_node.node_id]

        return branch_decision_node_list

    def _set_branch_decision_node_dict(self):
        branch_decision_node_dict = dict()

        decision_path_list = self.boosting_tree_interpreter.decision_path_list
        leaf_node_list = self.boosting_tree_interpreter.boosting_tree.leaf_node_list

        for decision_path in decision_path_list:
            similar_decision_path_list = self._get_similar_decision_path(decision_path, leaf_node_list)

            self._get_branch_decision_node_list(decision_path, similar_decision_path_list, branch_decision_node_dict)

            return branch_decision_node_dict

    def _get_similar_decision_path(self, decision_path_of_prediction, leaf_node_list):
        decision_path_list = list()

        margin_of_prediction = decision_path_of_prediction.decision_node_list[-1].margin

        for index, leaf_node in enumerate(leaf_node_list):
            if (leaf_node.margin > 0) and (leaf_node.margin > margin_of_prediction):
                decision_path_list.append(self.boosting_tree_interpreter.decision_path_list[index])

        return decision_path_list

    def _get_branch_decision_node_list(self, decision_path_of_prediction, decision_path_of_tree_list, branch_decision_node_dict):
        key_of_branch_decision_node = decision_path_of_prediction.decision_node_list[-1].node_id

        for decision_path_of_tree in decision_path_of_tree_list:
            decision_node = self._get_branch_decision_node(decision_path_of_prediction, decision_path_of_tree)
            if decision_node is None:
                raise Exception("Failed to found to interpret node.")

            branch_decision_node = BranchDecisionNode(branch_decision_node=decision_node,
                                                      decision_path_of_similarity=decision_path_of_tree,
                                                      decision_path_of_prediction=decision_path_of_prediction,
                                                      boosting_round_of_tree=self.boosting_tree_interpreter.boosting_tree.boosting_round,
                                                      decision_weight=decision_path_of_tree.decision_node_list[-1].margin,
                                                      variable_name=self.boosting_tree_interpreter.variable_name_list[decision_node.feature_index])

            if key_of_branch_decision_node not in branch_decision_node_dict.keys():
                branch_decision_node_dict[key_of_branch_decision_node] = [branch_decision_node]
            else:
                branch_decision_node_dict[key_of_branch_decision_node].append(branch_decision_node)

    def _get_branch_decision_node(self, decision_path_of_prediction, decision_path_of_tree):
        branch_decision_node = None

        for decision_node_of_prediction, decision_node_of_tree in zip(decision_path_of_prediction.decision_node_list, decision_path_of_tree.decision_node_list):
            if decision_node_of_prediction.node_id != decision_node_of_tree.node_id:
                break

            branch_decision_node = decision_node_of_tree

        return branch_decision_node
