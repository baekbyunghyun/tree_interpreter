import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from lifelog_analysis.data.ml_dataset.data_reader import DataReader
from lifelog_analysis.service.lifestyle_guide.interpret.random_forest_interpreter import RandomForestInterpreter
from lifelog_analysis.service.feature.health_calorie_guide_feature_metadata import HealthCalorieGuideFeatureMetadata
from lifelog_analysis.service.lifestyle_guide.calorie_health_guide import CalorieHealthGuide


def get_classifier(dataset=None):
    classifier = RandomForestClassifier(n_estimators=100, bootstrap=True, max_samples=0.6, max_depth=None, criterion='gini', max_features='auto')
    classifier.fit(dataset.X, dataset.y)

    return classifier


FEATURE_DIR_PATH = r'D:\vitameans_workspace\ai-modeling\workspace\feature\HealthCalorieGuideFeature'
TRAIN_FEATURE_FILE_NAME = r'users.spa'
ROOT_RESULT_DIR_PATH = r'D:\vitameans_workspace\ai-modeling\workspace\service\health_calorie_guide'
FEATURE_METADATA = HealthCalorieGuideFeatureMetadata()
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
NUMBER_OF_GUIDE = 10


if __name__ == '__main__':
    feature_file_path = os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)

    train_dataset = DataReader(os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)).get_dataset()
    train_dataset.shuffle()

    interpreter = RandomForestInterpreter(classifier=get_classifier(dataset=train_dataset),
                                          variable_name_list=FEATURE_METADATA.VARIABLE_NAME_LIST,
                                          class_name_list=FEATURE_METADATA.CLASS_NAME_LIST)

    # step_health_guide = StepHealthGuide(interpreter=interpreter, feature_metadata=FEATURE_METADATA, use_invalid_guide_filter=False)
    # sleep_health_guide = SleepHealthGuide(interpreter=interpreter, feature_metadata=FEATURE_METADATA, use_invalid_guide_filter=False)
    calorie_health_guide = CalorieHealthGuide(interpreter=interpreter, feature_metadata=FEATURE_METADATA, use_invalid_guide_filter=False)

    result_str_list = list()
    for test_user_id in TEST_USER_ID_LIST:
        test_dataset = DataReader(os.path.join(FEATURE_DIR_PATH, 'user_{0}.spa'.format(test_user_id)), used_reset_class_label=False).get_dataset()

        # number_of_step_guide_list = list()
        # number_of_sleep_guide_list = list()
        number_of_calorie_guide_list = list()

        line = '[{0}]'.format(test_user_id)
        index = 0
        for x, y in zip(test_dataset.X, test_dataset.y):
            # step_guide_habit_list = step_health_guide.find(x=x, y=y, number_of_guide=NUMBER_OF_GUIDE)
            # sleep_guide_habit_list = sleep_health_guide.find(x=x, y=y, number_of_guide=NUMBER_OF_GUIDE)
            calorie_guide_habit_list = calorie_health_guide.find(x=x, y=y, number_of_guide=NUMBER_OF_GUIDE)

            # number_of_step_guide_list.append(len(step_guide_habit_list))
            # number_of_sleep_guide_list.append(len(sleep_guide_habit_list))
            number_of_calorie_guide_list.append(len(calorie_guide_habit_list))

            line += 'Lifelog #{0}-{1}\n'.format(index, FEATURE_METADATA.CLASS_NAME_LIST[int(y)])
            # line += 'Number of step guide: {0}\n'.format(len(step_guide_habit_list))
            # line += 'Number of sleep guide: {0}\n'.format(len(sleep_guide_habit_list))
            line += 'Number of calorie guide: {0}\n\n'.format(len(calorie_guide_habit_list))

            index += 1

        # line += '[Step] Mean: {0}, Min: {1}, Max: {2}\n'.format(np.mean(number_of_step_guide_list), np.min(number_of_step_guide_list), np.max(number_of_step_guide_list))
        # line += '[Sleep] Mean: {0}, Min: {1}, Max: {2}\n'.format(np.mean(number_of_sleep_guide_list), np.min(number_of_sleep_guide_list), np.max(number_of_sleep_guide_list))
        line += '[Calorie] Mean: {0}, Min: {1}, Max: {2}\n\n'.format(np.mean(number_of_calorie_guide_list), np.min(number_of_calorie_guide_list), np.max(number_of_calorie_guide_list))

        result_str_list.append(line)

    for line in result_str_list:
        print(line)
