# ClearML Serving preprocessing code for the model of pytorch_mnist_serving_options.py
#
# Stored with the model by OutputModel.set_serving_preprocess. ClearML Serving runs it around the model:
# preprocess() turns a request into the tensor the model takes, postprocess() turns the model's output
# into the answer.
from typing import Any

import numpy as np


# Notice: the class must be named "Preprocess"
class Preprocess(object):
    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        # the request: {"pixels": 28 rows of 28 values, 0 (black) to 255 (white)}
        pixels = np.asarray(body["pixels"], dtype=np.float32).reshape(1, 28, 28)
        # the scaling and normalization the model was trained with
        return (pixels / 255.0 - 0.1307) / 0.3081

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # the model answers one score per digit: return the most likely one
        return {"digit": int(np.asarray(data).reshape(-1, 10)[0].argmax())}
