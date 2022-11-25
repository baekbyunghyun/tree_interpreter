import numpy as np


class InterpretedVariable:
    def __init__(self):
        self.variable_name = None
        self.value_list = list()
        self.similarity_list = list()
        self.threshold_list = list()

    @staticmethod
    def generate(variable_name, value, similarity, threshold):
        interpreted_variable = InterpretedVariable()
        interpreted_variable.variable_name = variable_name
        interpreted_variable.value_list.append(value)
        interpreted_variable.similarity_list.append(similarity)
        interpreted_variable.threshold_list.append(threshold)

        return interpreted_variable

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = 'variable name,' + '{0}'.format(self.variable_name)
        msg += '\nvalues,' + '{0}'.format(self.value_list)
        msg += '\nmedian threshold,' + '{0}'.format(np.median(self.threshold_list))
        msg += '\nmean threshold,' + '{0}'.format(np.mean(self.threshold_list))
        msg += '\nmax threshold,' + '{0}'.format(np.max(self.threshold_list))
        msg += '\nmin threshold,' + '{0}'.format(np.min(self.threshold_list))
        msg += '\nmedian similar,' + '{0}'.format(np.median(self.similarity_list))
        msg += '\nmean similar,' + '{0}'.format(np.mean(self.similarity_list))
        msg += '\nmax similar,' + '{0}'.format(np.max(self.similarity_list))
        msg += '\nmin similar,' + '{0}'.format(np.min(self.similarity_list))

        return msg
