import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from dataset.data_reader import DataReader
from classifier.custom_xgboost_classifier import CustomXGBoostClassifier
from classifier.custom_lightgbm_classifier import CustomLightGBMClassifier
from interpret.random_forest_interpreter import RandomForestInterpreter
from interpret.lightgbm_interpreter import LightGBMInterpreter
from interpret.xgboost_interpreter import XGBoostInterpreter
from branch.life_habit_pool import LifeHabitPool
from lifelog_analysis.service.lifestyle_guide.filter.inactive_variable_filter import InActiveVariableFilter
from lifelog_analysis.service.lifestyle_guide.filter.step_variable_filter import StepVariableFilter
from lifelog_analysis.service.feature.health_step_guide_feature_metadata import HealthStepGuideFeatureMetadata


def get_weight_based_optimal_class_index(class_index):
    if class_index == 0:
        return 0

    if class_index == 1:
        return 1

    if class_index == 2:
        return 1

    if class_index == 3:
        return 2

    return 1


class StepHealthGuide:
    def __init__(self, interpreter=None, feature_metadata=None, optimal_class_find_func=None, use_invalid_guide_filter=False):
        if interpreter is None:
            raise Exception('Invalid parameter for interpreter')

        if feature_metadata is None:
            raise Exception('Invalid parameter for feature metadata')

        if optimal_class_find_func is None:
            raise Exception('Invalid parameter for optimal class find function')

        self.interpreter = interpreter
        self.feature_metadata = feature_metadata
        self.optimal_class_find_func = optimal_class_find_func
        self.use_invalid_guide_filter = use_invalid_guide_filter

        self.inactive_filter = InActiveVariableFilter(self.feature_metadata.INACTIVE_VARIABLE_NAME_LIST)
        self.step_variable_filter = StepVariableFilter(self.feature_metadata.STEP_VARIABLE_NAME_LIST)

    def find(self, x=None, y=None, number_of_guide=3):
        life_habit_pool = LifeHabitPool.generate_life_habit_pool(interpreter=self.interpreter, x=x, y=y,
                                                                 optimal_class_find_func=self.optimal_class_find_func)

        guide_habit_list = list()
        for variable_name, life_habit in life_habit_pool.life_habit_dict.items():
            if self.inactive_filter.is_filtered_out(variable_name):
                continue

            if self.step_variable_filter.is_filtered_out(variable_name):
                continue

            if self._is_invalid_guide(life_habit):
                continue

            guide_habit_list.append(life_habit)

        if len(guide_habit_list) <= number_of_guide:
            return guide_habit_list

        return guide_habit_list[:number_of_guide]

    def _is_invalid_guide(self, life_habit):
        if not self.use_invalid_guide_filter:
            return False

        variable_name = life_habit.variable_name
        now = life_habit.value
        propose = np.median(life_habit.threshold_list)

        if (variable_name not in self.feature_metadata.BIGGER_GUIDE_STEP_VARIABLE_NAME_LIST) \
                and (variable_name not in self.feature_metadata.SMALLER_GUIDE_STEP_VARIABLE_NAME_LIST):
            return False

        if variable_name in self.feature_metadata.BIGGER_GUIDE_STEP_VARIABLE_NAME_LIST:
            if now <= propose:
                return False

        if variable_name in self.feature_metadata.SMALLER_GUIDE_STEP_VARIABLE_NAME_LIST:
            if now > propose:
                return False

        return True


def get_rf_classifier(dataset=None):
    classifier = RandomForestClassifier(n_estimators=51, max_depth=17, criterion='entropy', max_features=None)
    classifier.fit(dataset.X, dataset.y)

    return classifier


def get_xg_classifier(dataset=None):
    classifier = CustomXGBoostClassifier(n_estimators=100, max_depth=21, n_jobs=3, learning_rate=0.25, booster="gbtree", objective="multi:softmax")
    classifier.fit(dataset.X, dataset.y)

    return classifier


def get_lightgbm_classifier(dataset=None):
    classifier = CustomLightGBMClassifier(n_estimators=100, max_depth=21, n_jobs=3, objective="softmax")
    classifier.fit(dataset.X, dataset.y)

    return classifier


FEATURE_DIR_PATH = r'D:\givita-ai\ai-modeling\workspace\feature\HealthStepGuideFeature'
TRAIN_FEATURE_FILE_NAME = r'female_users.spa'
ROOT_RESULT_DIR_PATH = r'D:\givita-ai\ai-modeling\workspace/service/health_step_guide'
FEATURE_METADATA = HealthStepGuideFeatureMetadata()
TEST_USER_ID_LIST = [509, 510, 511, 513, 515,
                     661, 663, 664, 666, 667,
                     737, 738, 739, 740, 742,
                     883, 884, 885, 886, 887,
                     ]
NUMBER_OF_GUIDE = 10


if __name__ == '__main__':
    feature_file_path = os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)

    train_dataset = DataReader(os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)).get_dataset()
    train_dataset.shuffle()

    rf_classifier = get_rf_classifier(train_dataset)
    interpreter = RandomForestInterpreter(classifier=rf_classifier,
                                          variable_name_list=FEATURE_METADATA.VARIABLE_NAME_LIST,
                                          class_name_list=FEATURE_METADATA.CLASS_NAME_LIST)

    # xg_classifier = get_xg_classifier(train_dataset)
    # interpreter = XGBoostInterpreter(classifier=xg_classifier,
    #                                  variable_name_list=FEATURE_METADATA.VARIABLE_NAME_LIST,
    #                                  class_name_list=FEATURE_METADATA.CLASS_NAME_LIST)

    # lightgbm_classifier = get_lightgbm_classifier(train_dataset)
    # interpreter = LightGBMInterpreter(classifier=lightgbm_classifier,
    #                                   variable_name_list=FEATURE_METADATA.VARIABLE_NAME_LIST,
    #                                            class_name_list=FEATURE_METADATA.CLASS_NAME_LIST)

    step_health_guide = StepHealthGuide(interpreter=interpreter,
                                        feature_metadata=FEATURE_METADATA,
                                        optimal_class_find_func=get_weight_based_optimal_class_index,
                                        use_invalid_guide_filter=True)

    for test_user_id in TEST_USER_ID_LIST:
        test_dataset = DataReader(os.path.join(FEATURE_DIR_PATH, 'user_{0}.spa'.format(test_user_id)), used_reset_class_label=False).get_dataset()

        lines = ''
        index = 0
        for x, y in zip(test_dataset.X, test_dataset.y):
            guide_habit_list = step_health_guide.find(x=x, y=y, number_of_guide=NUMBER_OF_GUIDE)

            lines += '\n\n[Lifelog #{0} - {1}'.format(index, FEATURE_METADATA.CLASS_NAME_LIST[int(y)])
            for habit in guide_habit_list:
                lines += '\n{0}'.format(habit)

            index += 1

        result_file_path = os.path.join(ROOT_RESULT_DIR_PATH, 'step_health_guide_for_user_{0}_using_rf.csv'.format(test_user_id))
        with open(result_file_path, 'w') as file_descriptor:
            file_descriptor.write(lines)
