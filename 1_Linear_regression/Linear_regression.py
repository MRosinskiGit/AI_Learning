import threading

import matplotlib

from src.lib_linear_regression import RegressionOwn
from src.utils import config_logger

matplotlib.use("TkAgg")


def main():
    config_logger()
    dataset_hash = "burnoutminer/heights-and-weights-dataset"

    # Own Analyze
    Reg = RegressionOwn()
    path_to_dataset_dir = Reg.download_dataset(dataset_hash)
    file = Reg.find_csv_file_in_directory(path_to_dataset_dir)
    Reg.df = Reg.open_csv(file)
    Reg.recalculate_to_si()
    Reg.normalize_values()
    a_norm, b_norm = Reg.find_extremum_recurrence_own(0, 0, 0.1, 100)
    a_own, b_own = Reg.recalculate_linear_equation(a_norm, b_norm)
    a_sci, b_sci = Reg.use_scikit_regression("Height", "Weight")
    thread = threading.Thread(target=lambda: Reg.visualize_static(x="Height", y="Weight", linear=((a_own, b_own),)))
    thread.start()


main()
