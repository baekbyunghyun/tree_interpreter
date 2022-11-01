import numpy as np


class BoostingDecisionNode:
    DECISION_NODE_REGULAR_EXPRESSION = r"(\d*):(\[f(\d*)<([+-]?\d*\.?\d*[Ee]?[+-]?\d*)\]\s)?(leaf=([+-]?\d*\.?\d*" \
                                       r"[Ee]?[+-]?\d*))?(yes=(\d*),no=(\d*),missing=(\d*),gain=([+-]?\d*\.?\d*[Ee]?" \
                                       r"[+-]?\d*))?,cover=([+-]?\d*\.?\d*[Ee]?[+-]?\d*)"

    def __init__(self):
        self.node_id = None
        self.feature_index = None
        self.threshold = None
        self.gain = None
        self.cover = None
        self.margin = None
        self.weight = None                  # for lightgbm
        self.number_of_train_data = None    # for lightgbm
        self.yes_node_id = None
        self.no_node_id = None
        self.missing_node_id = None

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = '------------ Information of Decision {0}th node ------------'.format(self.node_id)
        msg += "\n Node id:                  {0}".format(self.node_id)
        msg += "\n Feature document:         {0}".format(self.feature_index)
        msg += "\n Threshold:                {0}".format(self.threshold)
        msg += "\n Gain:                     {0}".format(self.gain)
        msg += "\n Cover:                    {0}".format(self.cover)
        msg += "\n Margin:                   {0}".format(self.margin)
        msg += "\n Weight:                   {0}".format(self.weight)
        msg += "\n Number of train dataset:     {0}".format(self.number_of_train_data)
        msg += "\n Yes node id:              {0}".format(self.yes_node_id)
        msg += "\n No node id:               {0}".format(self.no_node_id)
        msg += "\n Missing node id:          {0}".format(self.missing_node_id)
        msg += "\n\n"

        return msg
