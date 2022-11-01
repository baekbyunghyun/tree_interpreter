import lightgbm
import numpy as np


from interpret.tree.boosting_tree import BoostingTree
from utils.utils import sigmoid


class CustomLightGBMClassifier(lightgbm.LGBMClassifier):
    CLASSIFIER_NAME = "lightgbm"

    def __init__(self, n_estimators=10, max_depth=None, objective="binary", random_state=None, **kwargs):
        super(CustomLightGBMClassifier, self).__init__(n_estimators=n_estimators, max_depth=max_depth, objective=objective, random_state=random_state, **kwargs)

        self.boosting_tree_list = None

    def fit(self, X, y, **kwargs):
        super(CustomLightGBMClassifier, self).fit(X, y, **kwargs)

        self._set_boosting_tree_info()

    def _set_boosting_tree_info(self):
        model_dump = self.booster_.dump_model(num_iteration=0)
        tree_dump_list = model_dump["tree_info"]

        self.boosting_tree_list = list()
        for index, tree_dump in enumerate(tree_dump_list):
            boosting_tree = BoostingTree.generate(index=index,
                                                  number_of_classes=self.n_classes_,
                                                  tree_dump=tree_dump,
                                                  type_of_classifier=self.CLASSIFIER_NAME)

            self.boosting_tree_list.append(boosting_tree)

    def predict(self, X, **kwargs):
        return super(CustomLightGBMClassifier, self).predict(X, **kwargs)

    def predict_one_vs_rest(self, X, class_of_boosting_tree=None):
        if not self.__sklearn_is_fitted__():
            raise Exception("Classifier is not trained.")

        y_pred_list = list()

        output_margins = self.predict(X, raw_score=True)

        if self.n_classes_ == 2:
            for output_margin in output_margins:
                if sigmoid(output_margin) > 0.5:
                    y_pred_list.append(1)

                else:
                    y_pred_list.append(0)

        else:
            if class_of_boosting_tree is None:
                raise Exception("Invalid class of boosting tree.")

            for output_margin in output_margins:
                if sigmoid(output_margin[class_of_boosting_tree]) > 0.5:
                    y_pred_list.append(1)

                else:
                    y_pred_list.append(0)

        return y_pred_list

    def predict_margin_by_tree(self, X, class_of_boosting_tree):
        if not self.__sklearn_is_fitted__():
            raise Exception("Classifier is not trained.")

        predicted_margin_list = list()
        for index in range(class_of_boosting_tree, len(self.boosting_tree_list), self.n_classes_):
            boosting_tree = self.boosting_tree_list[index]

            predicted_margin_list.append(boosting_tree.predict(X))

        predicted_margin_list = np.array(predicted_margin_list).T

        return predicted_margin_list


if __name__ == "__main__":
    import os

    from lifelog_analysis.data.ml_dataset.data_reader import DataReader

    from sklearn.model_selection import StratifiedKFold

    FEATURE_DIR_PATH = "/Users/youngho/Documents/GI-VITA/gi-vita/resource/feature"
    TRAIN_FEATURE_FILE_NAME = "220620_mission_guide_feature_delay_target.spa"

    feature_file_path = os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)

    dataset = DataReader(os.path.join(FEATURE_DIR_PATH, TRAIN_FEATURE_FILE_NAME)).get_dataset()

    accuracy_trains = list()
    precision_trains = list()
    recall_trains = list()
    f1score_trains = list()

    accuracy_tests = list()
    precision_tests = list()
    recall_tests = list()
    f1score_tests = list()

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=4570)
    for train_indexes, test_indexes in kf.split(dataset.X, dataset.y):
        X_train = dataset.X[train_indexes]
        y_train = dataset.y[train_indexes]
        X_test = dataset.X[test_indexes]
        y_test = dataset.y[test_indexes]

        model = CustomLightGBMClassifier(n_estimators=100, max_depth=21)
        model.fit(X_train, y_train)

        pred = np.zeros(X_train.shape[0])
        for tree in model.boosting_tree_list:
            pred += tree.predict(X_train)

        print(pred)
        print(model.predict(X_train, raw_score=True))
        # print(model.score(X_test, y_test))

        break
