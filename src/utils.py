from loguru import logger
import sys


import matplotlib

matplotlib.use("TkAgg")


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
