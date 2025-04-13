from loguru import logger
import threading


from src.lib_linear_regression import RegressionGradient
from src.utils import config_logger


def main():
    logger.info("Starting searching minimum by gradient.")
    config_logger()
    dataset_hash = "burnoutminer/heights-and-weights-dataset"

    Reg = RegressionGradient()
    path_to_dataset_dir = Reg.download_dataset(dataset_hash)
    file = Reg.find_csv_file_in_directory(path_to_dataset_dir)
    Reg.df = Reg.open_csv(file)
    Reg.recalculate_to_si()
    Reg.normalize_values()

    Reg.use_scikit_regression("Height", "Weight")
    a_sci_norm, b_sci_norm = Reg.use_scikit_regression("Height_std", "Weight_std")

    thread = threading.Thread(
        target=lambda: Reg.visualize_animated(x="Height_std", y="Weight_std", linear=((a_sci_norm, b_sci_norm),))
    )
    thread.start()

    a_norm, b_norm = Reg.find_extremum_recurrence_gradient(0, 0, learning_rate=0.7, iterations_lim=200, slowing=0.05)
    Reg.recalculate_linear_equation(a_norm, b_norm)


main()
