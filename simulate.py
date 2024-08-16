"""
Simulate Prediction Module

This module provides a class for generating and processing simulation data
for prediction tasks, particularly useful for testing domain adaptation algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SimulatePrediction:
    """
    A class to simulate prediction scenarios with domain adaptation.

    Attributes:
        A, B, C (float): Coefficients for source domain function.
        a, b, c (float): Coefficients for target domain function.
    """

    def __init__(self, A=-6, B=0, C=1, a=1, b=1, c=1):
        """
        Initialize the SimulatePrediction object.

        Args:
            A, B, C (float): Coefficients for source domain function.
            a, b, c (float): Coefficients for target domain function.
        """
        self.A, self.B, self.C = A, B, C
        self.a, self.b, self.c = a, b, c

    def data_generation(self, r1=1, r2=2, r3=3, source_num=500, target_num=10, test_num=200,
                        source_loc=5, source_scale=3, source_noise_scale=200,
                        target_uniform=5, target_noise_scale=12):
        """
        Generate synthetic data for source, target, and test sets.

        Args:
            r1, r2, r3 (int): Random seeds for source, target, and test data generation.
            source_num (int): Number of source domain samples.
            target_num (int): Number of target domain samples.
            test_num (int): Number of test samples.
            source_loc (float): Mean of source domain distribution.
            source_scale (float): Standard deviation of source domain distribution.
            source_noise_scale (float): Noise scale for source domain.
            target_uniform (float): Range for uniform distribution in target domain.
            target_noise_scale (float): Noise scale for target domain.
        """
        self.r1, self.r2, self.r3 = r1, r2, r3

        # Generate source domain data
        rng1 = np.random.RandomState(self.r1)
        self.x_source = rng1.normal(loc=source_loc, scale=source_scale, size=(source_num, 1))
        self.y_source = self.A * self.x_source + self.B * np.power(self.x_source, 2) + self.C * np.power(self.x_source, 3)
        self.y_source += rng1.normal(0, source_noise_scale, size=self.y_source.shape)

        # Generate target domain data
        rng2 = np.random.RandomState(self.r2)
        self.x_target = rng2.uniform(-target_uniform, target_uniform, size=(target_num, 1))
        self.y_target = self.a * self.x_target + self.b * np.power(self.x_target, 2) + self.c * np.power(self.x_target, 3)
        self.y_target += rng2.normal(0, target_noise_scale, size=self.y_target.shape)

        # Generate test data
        rng3 = np.random.RandomState(self.r3)
        self.x_test = rng3.uniform(-target_uniform, target_uniform, size=(test_num, 1))
        self.y_test = self.a * self.x_test + self.b * np.power(self.x_test, 2) + self.c * np.power(self.x_test, 3)
        self.y_test += rng3.normal(0, target_noise_scale, size=self.y_test.shape)

        # Load target data from Excel file (if needed)
        path_writer = "baseline_simulate.xlsx"
        sheet = 'plot'
        self.x_target = pd.read_excel(path_writer, sheet_name=sheet).values[:, 5:6]
        self.y_target = pd.read_excel(path_writer, sheet_name=sheet).values[:, 6:7]

    def data_normalization(self):
        """
        Normalize the generated data using StandardScaler and save to Excel.
        """
        self.std1, self.std2, self.std3, self.std4 = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()

        # Store original data
        data_source = np.hstack((self.x_source, self.y_source))
        data_target = np.hstack((self.x_target, self.y_target))

        # Normalize data
        self.x_source = self.std1.fit_transform(self.x_source)
        self.y_source = self.std2.fit_transform(self.y_source)
        self.x_target = self.std3.fit_transform(self.x_target)
        self.y_target = self.std4.fit_transform(self.y_target)

        self.x_sourcetarget = np.vstack((self.x_source, self.x_target))
        self.y_sourcetarget = np.vstack((self.y_source, self.y_target))

        x_test_real = self.x_test
        y_test_real = self.a * x_test_real + self.b * np.power(x_test_real, 2) + self.c * np.power(x_test_real, 3)
        self.x_test = self.std3.transform(self.x_test)

        # Prepare data for saving
        data_frames = {
            'source_normal': pd.DataFrame(np.hstack((self.x_source, self.y_source)),
                                          columns=['x_source_normal', 'y_source_normal']),
            'target_normal': pd.DataFrame(np.hstack((self.x_target, self.y_target)),
                                          columns=['x_target_normal', 'y_target_normal']),
            'test_real': pd.DataFrame(np.hstack((x_test_real, self.y_test, y_test_real)),
                                      columns=['x_test_real', 'y_test_true', 'y_test_real']),
            'source_real': pd.DataFrame(data_source, columns=['x_source', 'y_source']),
            'target_real': pd.DataFrame(data_target, columns=['x_target', 'y_target'])
        }

        # Save to Excel
        with pd.ExcelWriter("original_datanew.xlsx") as writer:
            for sheet_name, df in data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    simulate_process = SimulatePrediction(A=-5, C=2)
    simulate_process.data_generation(r1=7, source_num=500, test_num=200, source_loc=7, source_scale=3,
                                     target_uniform=4, source_noise_scale=200, target_noise_scale=6, target_num=10)
    simulate_process.data_normalization()