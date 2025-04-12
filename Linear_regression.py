import sys
import time

import kagglehub
from loguru import logger
import numpy as np
import pandas as pd
import os
import threading

import matplotlib
from sklearn.linear_model import LinearRegression

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def recalculate_in_to_cm(inch):
    return inch * 2.54


def recalculate_pound_to_kg(pound):
    return pound * 0.45359237


def calculate_y(x, a_param, b_param):
    return a_param * x + b_param


def config_logger():
    logger.remove()
    logger.add(
        "logs/log_{time:YYYY-MM-DD_HH-mm-ss}.log",
        rotation="5 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {module}.{function}:{line} | {message}",
        enqueue=True,
    )
    logger.add(sys.stdout, level="INFO", colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}")


def download_dataset(dataset_hash) -> str:
    logger.info(f"Downloading {dataset_hash} dataset.")
    path = kagglehub.dataset_download(dataset_hash)
    logger.success(f"Downloaded to path {path}")
    return path


def find_csv_file_in_directory(path):
    logger.info(f"Seraching path {path}")
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


def open_csv(path):
    with open(path) as f:
        logger.info(f"Loading csv file {path}")
        return pd.read_csv(f)


def calculate_rss(a, b, x="Height_std", y="Weight_std"):
    global df
    y_true = df[y]
    y_pred = calculate_y(df[x], a_param=a, b_param=b)
    return ((y_true - y_pred) ** 2).sum()


def find_extremum_reccurency(a, b, step, iterations_lim, current_iter=0):
    logger.info(f"Starting iteration with values: a:{a}, b:{b} | step: a:{step} | cuur_iter:{current_iter}")
    if current_iter == iterations_lim:
        logger.info("Reached limit of iterations")
        return (a, b)

    possibilites = lambda a, b: [
        (a + step, b),
        (a + step, b + step),
        (a, b + step),
        (a - step, b),
        (a, b - step),
        (a - step, b - step),
        (a + step, b - step),
        (a - step, b + step),
    ]

    current_best_result = calculate_rss(a, b)
    current_best_combinaton = (a, b)
    tol = 1e-6
    improvement = True
    while improvement:
        enter_loop_data = current_best_combinaton
        all_possibilites = possibilites(*current_best_combinaton)
        for option in all_possibilites:
            rss = calculate_rss(*option)
            if rss < current_best_result:
                current_best_result = rss
                current_best_combinaton = option
        improvement = abs(calculate_rss(*enter_loop_data) - current_best_result) > tol
    return find_extremum_reccurency(*current_best_combinaton, step * 0.7, iterations_lim, current_iter + 1)


def visualize(**kwargs):
    global df
    plt.scatter(x=df[kwargs.get("x")], y=df[kwargs.get("y")], s=0.15)

    x_vals = np.linspace(df[kwargs.get("x")].min(), df[kwargs.get("x")].max(), 100)

    for a, b in kwargs.get("linear"):
        y_vals = a * x_vals + b
        plt.plot(x_vals, y_vals, linewidth=1)

    plt.grid(True)
    plt.show()


def recalculate_to_si():
    global df
    df = df.assign(Height=recalculate_in_to_cm(df["Height(Inches)"]))
    return df.assign(Weight=recalculate_pound_to_kg(df["Weight(Pounds)"]))


def normalize_values():
    global df
    df = df.assign(Height_std=lambda x: (df["Height"] - df["Height"].min()) / (df["Height"].max() - df["Height"].min()))
    return df.assign(
        Weight_std=lambda x: (df["Weight"] - df["Weight"].min()) / (df["Weight"].max() - df["Weight"].min())
    )


def recalculate_linear_equasion(a_norm, b_norm):
    global df
    y_max = df["Weight"].max()
    y_min = df["Weight"].min()
    x_max = df["Height"].max()
    x_min = df["Height"].min()
    a = (a_norm * y_max - a_norm * y_min) / (x_max - x_min)
    b = -a * x_min + b_norm * (y_max - y_min) + y_min
    logger.info(f"Parameters recalculated from normalized to std: a:{a}, b:{b}")
    return a, b


def main():
    global df
    config_logger()
    dataset_hash = "burnoutminer/heights-and-weights-dataset"

    # Own Analyze
    path_to_dataset_dir = download_dataset(dataset_hash)
    file = find_csv_file_in_directory(path_to_dataset_dir)
    df = open_csv(file)
    df = recalculate_to_si()
    df = normalize_values()
    a_norm, b_norm = find_extremum_reccurency(0, 0, step=0.1, iterations_lim=100)
    a_own, b_own = recalculate_linear_equasion(a_norm, b_norm)

    # Scikitlearn
    X = df[["Height"]]
    y = df["Weight"]
    model = LinearRegression()
    model.fit(X, y)
    a2 = model.coef_[0]
    b2 = model.intercept_
    logger.info(f"Linear regression from scikit: Y = {a2:.4f} * X + {b2:.4f}")

    visualize_thread = threading.Thread(
        target=lambda: visualize(x="Height", y="Weight", linear=((a_own, b_own), (a2, b2)))
    )
    visualize_thread.start()


main()
