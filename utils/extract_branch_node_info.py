import os

from sklearn.ensemble import RandomForestClassifier

from dataset.data_reader import DataReader
from interpret.random_forest_interpreter import RandomForestInterpreter


FEATURE_DIR_PATH = r'/workspace/feature/WeightDerivationFeature'
TRAIN_FEATURE_FILE_NAME = r'users.spa'
ROOT_RESULT_DIR_PATH = r'/workspace/health_guide'
INACTIVE_VARIABLE_LIST = ['gender', 'age', 'height']
TEST_USER_ID_LIST = [229, 231, 232, 234, 235,
                     370, 371, 372, 373, 374,
                     435, 436, 437, 438, 439,
                     509, 510, 511, 513, 515,
                     661, 663, 664, 666, 667,
                     737, 738, 739, 740, 742,
                     883, 884, 885, 886, 887,
                     905, 906, 907, 908, 909,
                     1045, 1048, 1049, 1050, 1051,
                     1110, 1111, 1112, 1113, 1114]


def get_classifier(dataset=None):
    classifier = RandomForestClassifier(n_estimators=150, max_depth=13, criterion='entropy', max_features=None)
    classifier.fit(dataset.X, dataset.y)

    return classifier


def extract_life_pattern_info(interpreter=None, test_dataset=None, user_id=None, life_pattern_dir_path=None):
    user_directory = os.path.join(life_pattern_dir_path, '{0}'.format(user_id))
    if not os.path.exists(user_directory):
        os.mkdir(user_directory)

    for data_index in range(0, len(test_dataset.X)):
        x = test_dataset.X[data_index]
        y = test_dataset.y[data_index]

        analysis_info_str_list = list()

        branch_decision_nodes_per_estimator_dict = interpreter.find_branch_decision_node_per_estimator(x, y)
        for _, branch_decision_node_list in branch_decision_nodes_per_estimator_dict.items():
            branch_info_str_list = get_decision_node_info_per_decision_tree(branch_decision_node_list, x)

            analysis_info_str_list += branch_info_str_list

        if len(analysis_info_str_list) <= 0:
            continue

        result_file_path = os.path.join(user_directory, 'Lifelog #{0} class_{1}.csv'.format(data_index, y))
        with open(result_file_path, 'w') as file_descriptor:
            for analysis_info in analysis_info_str_list:
                file_descriptor.write(analysis_info)
                file_descriptor.write('\n')


def get_decision_node_info_per_decision_tree(branch_decision_node_list, x):
    branch_info_str_list = list()

    for index in range(0, len(branch_decision_node_list)):
        branch_decision_node = branch_decision_node_list[index]

        if branch_decision_node.variable_name in INACTIVE_VARIABLE_LIST:
            continue

        info_str = '{0},{1},{2},{3},{4}'.format('{0}'.format(index).center(10),
                                                '{0}'.format(branch_decision_node.similarity).ljust(10),
                                                branch_decision_node.variable_name.ljust(30),
                                                '{0}'.format(round(x[branch_decision_node.leaf_node.feature_index], 3)).ljust(20),
                                                '{0}'.format(round(branch_decision_node.leaf_node.threshold, 3)).ljust(20))

        branch_info_str_list.append(info_str)

    return branch_info_str_list


if __name__ == '__main__':
    class_name_list = ['UnderWeight', 'Normal', 'OverWeight', 'Obesity']

    feature_file_path = os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)

    train_dataset = DataReader(feature_file_path).get_dataset()
    train_dataset.shuffle()

    classifier = get_classifier(dataset=train_dataset)

    interpreter = RandomForestInterpreter(classifier=classifier,
                                          variable_name_list=variable_name_list,
                                          class_name_list=class_name_list)

    for user_id in TEST_USER_ID_LIST:
        feature_file_path = os.path.join(FEATURE_DIR_PATH, 'user_{0}.spa'.format(user_id))

        test_dataset = DataReader(feature_file_path, used_reset_class_label=False).get_dataset()
        test_dataset.shuffle()

        extract_life_pattern_info(interpreter=interpreter,
                                  test_dataset=test_dataset,
                                  user_id=user_id,
                                  life_pattern_dir_path=ROOT_RESULT_DIR_PATH)
