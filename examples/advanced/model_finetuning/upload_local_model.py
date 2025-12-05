import argparse
from clearml import OutputModel, Task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, help="Path to the model to be uploaded to ClearML"
    )
    parser.add_argument(
        "--model-name", type=str, help="Model name - to be displayed in the ClearML UI"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = Task.init(
        project_name="Example", task_name="Upload Local Model", output_uri=True
    )
    output_model = OutputModel(name=args.model_name)
    output_model.update_weights_package(
        weights_path=args.model_path, auto_delete_file=False
    )
