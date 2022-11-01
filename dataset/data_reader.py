import numpy as np

from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file

from .dataset import Dataset


class DataReader:
    def __init__(self, file_name=None):
        assert file_name is not None, 'File name is None.'

        self.file_name = file_name
        self.X = None
        self.y = None

        if Dataset.check_file_format(self.file_name) == Dataset.TYPE_DATASET_FORMAT_SPARSE:
            self._load_svm_light()
        elif Dataset.check_file_format(self.file_name) == Dataset.TYPE_DATASET_FORMAT_NORMAL:
            self._load_normal()
        else:
            raise Exception('Invalid file format.')

        self.dataset = Dataset(X=self.X, y=self.y)

    def _load_svm_light(self):
        X, self.y = load_svmlight_file(self.file_name)

        self.X = np.array(coo_matrix(X, dtype=np.float).todense())

    def _load_normal(self):
        X = np.loadtxt(self.file_name, delimiter=',')

        row_size, col_size = X.shape
        self.X = X[:, 1:col_size]
        self.y = X[:, 0]

    def get_dataset(self):
        return self.dataset
