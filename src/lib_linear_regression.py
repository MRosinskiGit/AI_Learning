import time
from abc import ABC

import kagglehub
from loguru import logger
import numpy as np
import pandas as pd
import os

from matplotlib.animation import FuncAnimation
import matplotlib
from sklearn.linear_model import LinearRegression

from src.utils import calculate_y, recalculate_in_to_cm, recalculate_pound_to_kg

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class RegressionCore(ABC):
    def __init__(self):
        self.df = ...

    @staticmethod
    def download_dataset(dataset_hash) -> str:
        logger.info(f"Downloading {dataset_hash} dataset.")
        path = kagglehub.dataset_download(dataset_hash)
        logger.success(f"Downloaded to path {path}")
        return path

    @staticmethod
    def find_csv_file_in_directory(path):
        logger.info(f"Searching path {path}")
        all_files = [file for file in filter(lambda x: x.endswith(".csv"), os.listdir(path))]
        logger.debug(f"Files found: {all_files}")
        if len(all_files) == 0:
            logger.warning("No files found")
            return
        elif len(all_files) == 1:
            return os.path.join(path, all_files[0])
        else:
            time.sleep(0.5)
            print("Found multiple csv files. Select correct")
            for index, file in enumerate(all_files):
                print(f"\t {index + 1}: {file}")
            selected_file = input("Pick number.")
            return os.path.join(path, all_files[int(selected_file) - 1])

    @staticmethod
    def open_csv(path):
        with open(path) as f:
            logger.info(f"Loading csv file {path}")
            return pd.read_csv(f)

    def recalculate_to_si(self):
        self.df = self.df.assign(Height=recalculate_in_to_cm(self.df["Height(Inches)"]))
        self.df = self.df.assign(Weight=recalculate_pound_to_kg(self.df["Weight(Pounds)"]))

    def normalize_values(self):
        self.df = self.df.assign(
            Height_std=lambda x: (self.df["Height"] - self.df["Height"].min())
            / (self.df["Height"].max() - self.df["Height"].min())
        )
        self.df = self.df.assign(
            Weight_std=lambda x: (self.df["Weight"] - self.df["Weight"].min())
            / (self.df["Weight"].max() - self.df["Weight"].min())
        )

    def use_scikit_regression(self, x, y):
        logger.info(f"Searching linear fit for:{x} and {y} by Scikitlearn")
        x_df = self.df[[x]]
        y_df = self.df[y]
        model = LinearRegression()
        model.fit(x_df, y_df)
        a = model.coef_[0]
        b = model.intercept_
        logger.success(f"Parameters found by scikit: {a} * X + {b}")
        return a, b

    def recalculate_linear_equation(self, a_norm, b_norm):
        y_max = self.df["Weight"].max()
        y_min = self.df["Weight"].min()
        x_max = self.df["Height"].max()
        x_min = self.df["Height"].min()
        a = (a_norm * y_max - a_norm * y_min) / (x_max - x_min)
        b = -a * x_min + b_norm * (y_max - y_min) + y_min
        logger.info(f"Parameters recalculated from normalized to std: a:{a}, b:{b}")
        return a, b


class RegressionOwn(RegressionCore):
    def __init__(self):
        super().__init__()

    def calculate_rss(self, a, b, x="Height_std", y="Weight_std"):
        y_true = self.df[y]
        y_pred = calculate_y(self.df[x], a_param=a, b_param=b)
        return ((y_true - y_pred) ** 2).sum()

    def find_extremum_recurrence_own(self, a, b, step, iterations_lim, current_iter=0):
        logger.info(f"Starting iteration with values: a:{a}, b:{b} | step: a:{step} | curr_iter:{current_iter}")
        if current_iter == iterations_lim:
            logger.info("Reached limit of iterations")
            return a, b

        possibilities = lambda x_par, y_par: [
            (x_par + step, y_par),
            (x_par + step, y_par + step),
            (x_par, y_par + step),
            (x_par - step, y_par),
            (x_par, y_par - step),
            (x_par - step, y_par - step),
            (x_par + step, y_par - step),
            (x_par - step, y_par + step),
        ]

        current_best_result = self.calculate_rss(a, b)
        current_best_combination = (a, b)
        tol = 1e-6
        improvement = True
        while improvement:
            enter_loop_data = current_best_combination
            all_possibilities = possibilities(*current_best_combination)
            for option in all_possibilities:
                rss = self.calculate_rss(*option)
                if rss < current_best_result:
                    current_best_result = rss
                    current_best_combination = option
            improvement = abs(self.calculate_rss(*enter_loop_data) - current_best_result) > tol
        return self.find_extremum_recurrence_own(
            *current_best_combination, step * 0.7, iterations_lim, current_iter + 1
        )

    def visualize_static(self, **kwargs):
        plt.scatter(x=self.df[kwargs.get("x")], y=self.df[kwargs.get("y")], s=0.15)

        x_vals = np.linspace(self.df[kwargs.get("x")].min(), self.df[kwargs.get("x")].max(), 100)

        for a, b in kwargs.get("linear"):
            y_vals = a * x_vals + b
            plt.plot(x_vals, y_vals, linewidth=1)

        plt.grid(True)
        plt.show()


class RegressionGradient(RegressionCore):
    def __init__(self):
        super().__init__()
        self.a_vis = 0
        self.b_vis = 0

    def calculate_rss_derivative(self, a, b, d, x="Height_std", y="Weight_std"):
        assert d in ["a", "b"], "Derivative must be calculated for a or b"
        y_true = self.df[y]
        y_pred = calculate_y(self.df[x], a_param=a, b_param=b)
        if d == "a":
            return (-2 / len(self.df)) * (((y_true - y_pred) * self.df[x]).sum())
        return (-2 / len(self.df)) * ((y_true - y_pred).sum())

    def find_extremum_recurrence_gradient(self, a, b, learning_rate, iterations_lim, current_iter=0, slowing=0):
        if current_iter % 100 == 0:
            logger.info(
                f"Starting iteration with values: a:{a}, b:{b} | step: a:{learning_rate} | curr_iter:{current_iter}"
            )
        if current_iter == iterations_lim:
            logger.info("Reached limit of iterations")
            return a, b
        da = self.calculate_rss_derivative(a, b, d="a")
        db = self.calculate_rss_derivative(a, b, d="b")
        a = a - learning_rate * da
        b = b - learning_rate * db
        self.a_vis = a
        self.b_vis = b
        if slowing:
            time.sleep(slowing)
        return self.find_extremum_recurrence_gradient(a, b, learning_rate, iterations_lim, current_iter + 1, slowing)

    def visualize_animated(self, **kwargs):
        fig, ax = plt.subplots()
        scatter = ax.scatter(self.df[kwargs.get("x")], self.df[kwargs.get("y")], s=0.15)  # noqa: F841
        x_vals_sci = np.linspace(self.df[kwargs.get("x")].min(), self.df[kwargs.get("x")].max(), 100)

        for a, b in kwargs.get("linear"):
            y_vals_lin = a * x_vals_sci + b
            plt.plot(x_vals_sci, y_vals_lin, linewidth=1)

        (line,) = ax.plot([], [], color="red", linewidth=1.5)

        ax.set_xlim(self.df[kwargs.get("x")].min(), self.df[kwargs.get("x")].max())
        ax.set_ylim(self.df[kwargs.get("y")].min(), self.df[kwargs.get("y")].max())
        ax.grid(True)
        ax.set_xlabel(kwargs.get("x"))
        ax.set_ylabel(kwargs.get("y"))

        x_vals = np.linspace(self.df[kwargs.get("x")].min(), self.df[kwargs.get("x")].max(), 100)

        def update(frame):
            y_vals = self.a_vis * x_vals + self.b_vis
            line.set_data(x_vals, y_vals)
            return (line,)

        ani = FuncAnimation(fig, update, frames=np.arange(0, 1000), interval=50, blit=True)  # noqa: F841
        plt.show()
