"""
ClearML Pipeline – Dynamic Loop Example
========================================
Demonstrates the new ``loop_condition`` / ``loop_body`` feature that
allows a pipeline step to conditionally **re-execute** a set of body
steps, achieving Prefect-style dynamic loops inside a ClearML DAG.

Pipeline structure::

    data_prep  ──>  train  ──>  evaluate  ──>  loop_ctrl  ──>  deploy
                      ^                           │
                      └───── (loop if acc < 95) ──┘

Usage (local debug)::

    python pipeline_with_dynamic_loop.py

Usage (remote)::

    python pipeline_with_dynamic_loop.py --run-remote
"""

import argparse
import random
import sys

from clearml import Task
from clearml.automation.controller import PipelineController


# ── Pipeline step functions ─────────────────────────────────────────

def step_data_prep(dataset_name="demo"):
    print("Preparing dataset '{}'".format(dataset_name))
    return dataset_name


def step_train(dataset="demo", lr=0.01):
    import random as _rnd
    acc = round(_rnd.uniform(85, 99), 2)
    print("Training on '{}' (lr={}) -> accuracy={}%".format(dataset, lr, acc))
    Task.current_task().get_logger().report_scalar("Accuracy", "train", acc, 0)
    return {"accuracy": acc, "lr": lr}


def step_evaluate(train_result=None):
    acc = (train_result or {}).get("accuracy", 0.0)
    print("Evaluate -> accuracy={}%".format(acc))
    return {"accuracy": acc, "passed": acc >= 95.0}


def step_deploy(eval_result=None):
    print("Deploying model (accuracy={})".format(eval_result))
    return "deployed"


def step_loop_placeholder():
    return "loop_placeholder"


# ── Loop condition callback ─────────────────────────────────────────

def should_retrain(pipeline, node):
    """Return True to re-run the loop body (train -> evaluate)."""
    eval_node = pipeline.get_pipeline_dag().get("evaluate")
    if not eval_node or not eval_node.executed:
        return False
    try:
        t = Task.get_task(task_id=eval_node.executed)
        metrics = t.get_last_scalar_metrics()
        acc = metrics.get("Accuracy", {}).get("train", {}).get("last", 0)
    except Exception:
        acc = 0.0
    should_loop = acc < 95.0
    print(
        "Loop condition: accuracy={:.2f}% {} (iteration {})".format(
            acc, "< 95% -> retrain" if should_loop else ">= 95% -> done",
            node._loop_iteration
        )
    )
    return should_loop


# ── Pipeline definition ─────────────────────────────────────────────

def build_pipeline(run_remote=False):
    pipe = PipelineController(
        name="Dynamic Loop Pipeline",
        project="examples",
        version="1.0.0",
    )

    pipe.add_function_step(
        name="data_prep",
        function=step_data_prep,
        function_kwargs=dict(dataset_name="demo"),
        function_return=["dataset"],
        cache_executed_step=False,
    )

    pipe.add_function_step(
        name="train",
        function=step_train,
        function_kwargs=dict(dataset="${data_prep.dataset}", lr=0.01),
        function_return=["train_result"],
        parents=["data_prep"],
        cache_executed_step=False,
    )

    pipe.add_function_step(
        name="evaluate",
        function=step_evaluate,
        function_kwargs=dict(train_result="${train.train_result}"),
        function_return=["eval_result"],
        parents=["train"],
        cache_executed_step=False,
    )

    pipe.add_function_step(
        name="loop_ctrl",
        function=step_loop_placeholder,
        function_return=["_loop_out"],
        parents=["evaluate"],
        loop_condition=should_retrain,
        loop_body=["train", "evaluate"],
        max_loop_iterations=5,
        cache_executed_step=False,
    )

    pipe.add_function_step(
        name="deploy",
        function=step_deploy,
        function_kwargs=dict(eval_result="${evaluate.eval_result}"),
        function_return=["deploy_status"],
        parents=["loop_ctrl"],
        cache_executed_step=False,
    )

    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-remote", action="store_true")
    args = parser.parse_args()

    pipe = build_pipeline(run_remote=args.run_remote)

    if args.run_remote:
        pipe.start(queue="default", wait=True)
    else:
        pipe.start_locally(run_pipeline_steps_locally=True)

    print("Pipeline finished!")


if __name__ == "__main__":
    main()
