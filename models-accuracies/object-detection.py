import os
import sys
import click

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))
from experiments.utils.constants import ACCURACIES_EVAL


@click.command()
@click.option("--model", required=True, type=str, default="yolov5n")
@click.option("--dataset", required=True, type=str, default="coco128")
def yolo_eval(model, dataset):
    """ """
    results_folder = os.path.join(ACCURACIES_EVAL, "yolo")
    command = (
        f"python ~/yolov5/val.py --weights {model}.pt --batch-size 16"
        f" --data {dataset}.yaml --img 640 --conf 0.001"
        f" --iou 0.65 --project {results_folder}"
    )
    os.system(command)


if __name__ == "__main__":
    yolo_eval()
