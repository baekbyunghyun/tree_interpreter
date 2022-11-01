import numpy as np

from .dataset import Dataset


class RandomDataGenerator:
    def __init__(self, file_name=None, num_of_data_per_class=None, col_size=None):
        assert file_name is not None and num_of_data_per_class is not None and col_size is not None, "Invalid argument."

        self.file_name = file_name
        self.num_of_data_per_class = num_of_data_per_class
        self.col_size = col_size
        self.row_size = np.sum(self.num_of_data_per_class)
        self.num_of_class = len(self.num_of_data_per_class)
        self.classes = np.arange(self.num_of_class)

        self.X = np.zeros((self.row_size, self.col_size))
        self.y = np.zeros(self.row_size)

        self.dataset = None

    def generate_2d_3c(self):
        # class 0
        total_count = self.num_of_data_per_class[0]
        count = 0
        self.y[count:total_count] = self.classes[0]
        while count < total_count:
            x = np.random.rand(self.col_size)
            if not ((0.2 < x[0] < 0.8) and (0.2 < x[1] < 0.8)):
                self.X[count] = x
                count = count + 1

        # class 1
        total_count = total_count + self.num_of_data_per_class[1]
        self.y[count:total_count] = self.classes[1]
        while count < total_count:
            x = np.random.rand(self.col_size)
            if (0.2 < x[0] < 0.8) and (0.2 < x[1] < 0.8) and (x[0] < x[1]):
                self.X[count] = x
                count = count + 1

        # class 2
        total_count = total_count + self.num_of_data_per_class[2]
        self.y[count:total_count] = self.classes[2]
        while count < total_count:
            x = np.random.rand(self.col_size)
            if (0.2 < x[0] < 0.8) and (0.2 < x[1] < 0.8) and (x[0] > x[1]):
                self.X[count] = x
                count = count + 1

        self.dataset = Dataset(X=self.X, y=self.y)
        self.dataset.save_dataset(self.file_name)

    def generate_2d_2c(self):
        # class 0
        total_count = self.num_of_data_per_class[0]
        count = 0
        self.y[count:total_count] = self.classes[0]
        while count < total_count:
            x = np.random.rand(self.col_size) * 3
            if (0.2 < x[0] < 1.5) and (1.3 < x[1] < 2.8):
                self.X[count] = x
                count = count + 1

        # class 1
        total_count = total_count + self.num_of_data_per_class[1]
        self.y[count:total_count] = self.classes[1]
        while count < total_count:
            x = np.random.rand(self.col_size) * 3
            if (1 < x[0] < 2.7) and (0.2 < x[1] < 2.3) and (x[0] > x[1]):
                self.X[count] = x
                count = count + 1

        self.dataset = Dataset(X=self.X, y=self.y)
        self.dataset.save_dataset(self.file_name)

    def get_dataset(self):
        return self.dataset
