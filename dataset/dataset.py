import copy
from random import randint
import numpy as np

from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


class Dataset:
    TYPE_DATASET_FORMAT_SPARSE = 0
    TYPE_DATASET_FORMAT_NORMAL = 1

    def __init__(self, X=None, y=None):
        assert (X is not None and y is not None), 'X or y is None.'

        self.X = X
        self.y = y
        self.index = 0

        self.row_size = None
        self.col_size = None
        self.num_of_classes = None
        self.num_of_data_per_class = None
        self.classes = None
        self.min_before_scale = None
        self.max_before_scale = None

        self.normalized = False

        self._init_parameter()

    def _init_parameter(self):
        self.row_size, self.col_size = self.X.shape
        self._reset_class_label()

        self.classes = np.unique(self.y)
        self.num_of_classes = len(self.classes)

        self._set_num_of_data_per_class()

    def _reset_class_label(self):
        copy_y = copy.deepcopy(self.y)
        target_y = np.unique(copy_y)

        self.y = np.zeros(self.row_size, dtype='int')

        for c, v in enumerate(target_y):
            indexes = np.where(copy_y == v)[0]
            self.y[indexes] = c

    def _set_num_of_data_per_class(self):
        self.num_of_data_per_class = np.zeros(self.num_of_classes)
        for i, v in enumerate(self.classes):
            self.num_of_data_per_class[i] = len(np.where(self.y == v)[0])

    def shuffle(self):
        shuffle_data = shuffle(self.X, self.y, random_state=randint(0, 1000))

        self.X = shuffle_data[0]
        self.y = shuffle_data[1]

    def scale(self):
        self.min_before_scale = np.min(self.X)
        self.max_before_scale = np.max(self.X)

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = scaler.fit_transform(self.X)

        self.normalized = True

    def save_dataset(self, file_name=None):
        assert file_name is not None, 'File name is None.'

        if Dataset.check_file_format(file_name) == Dataset.TYPE_DATASET_FORMAT_SPARSE:
            dump_svmlight_file(self.X, self.y, file_name, zero_based=False)

        else:
            data = np.column_stack((self.y, self.X))
            np.savetxt(file_name, data, fmt='%s', delimiter=',')

    def get_X_y(self):
        return self.X, self.y

    def get_index(self):
        return self.index

    def get_row_size(self):
        return self.row_size

    def get_col_size(self):
        return self.col_size

    def get_num_of_data_per_class(self, i):
        assert i <= 0 < self.num_of_classes, 'Out of range on class ' + str(i)

        return self.num_of_data_per_class[i]

    def get_num_of_classes(self):
        return self.num_of_classes

    def get_classes(self):
        return self.classes

    @staticmethod
    def check_file_format(file_name):
        if file_name.endswith('.spa'):
            return Dataset.TYPE_DATASET_FORMAT_SPARSE

        return Dataset.TYPE_DATASET_FORMAT_NORMAL

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = "------------ Overview on data ------------"
        msg += "\n data dimension    : {0} X {1}".format(self.row_size, self.col_size)
        msg += "\n no. of classes    : {0}".format(self.num_of_classes)
        msg += "\n data size per cls : {0}".format(self.num_of_data_per_class)
        msg += "\n class labels      : {0}".format(self.classes)
        msg += "\n normalized        : {0}".format(self.normalized)
        msg += "\n min, max          : {0}, {1}".format(np.min(self.X), np.max(self.X))
        msg += "\n (old min, old max): ({0}, {1})".format(self.min_before_scale, self.max_before_scale)
        msg += "\n mean              : {0}".format(np.mean(self.X, axis=0))
        msg += "\n variance          : {0}".format(np.var(self.X, axis=0))
        msg += "\n recommend radius  : {0}".format(0.3 * np.abs(np.max(self.X) - np.min(self.X)))
        msg += "\n---------------------------------------"

        return msg

    def __len__(self):
        return self.row_size

    def __getitem__(self, i):
        return i, self.y[i], self.X[i]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if 0 <= self.index < self.row_size:
            index = self.index
            self.index = self.index + 1

            return index, self.y[index], self.X[index]

        else:
            raise StopIteration()
