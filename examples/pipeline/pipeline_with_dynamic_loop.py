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

from __future__ import annotations

import argparse
import random
import sys

from clearml import Task
from clearml.automation.controller import PipelineController


# ── Pipeline step functions ─────────────────────────────────────────

def step_data_prep(dataset_name: str = "demo") -> str:
    print(f"Preparing dataset '{dataset_name}'")
    return dataset_name


def step_train(dataset: str = "demo", lr: float = 0.01) -> dict:
    acc = round(random.uniform(85, 99), 2)
    print(f"Training on '{dataset}' (lr={lr}) → accuracy={acc}%")
    Task.current_task().get_logger().report_scalar("Accuracy", "train", acc, 0)
    return {"accuracy": acc, "lr": lr}


def step_evaluate(train_result: dict = None) -> dict:
    acc = (train_result or {}).get("accuracy", 0.0)
    print(f"Evaluate → accuracy={acc}%")
    return {"accuracy": acc, "passed": acc >= 95.0}


def step_deploy(eval_result: dict = None) -> str:
    print(f"Deploying model (accuracy={eval_result})")
    return "deployed"


# ── Loop condition callback ─────────────────────────────────────────

def should_retrain(pipeline: PipelineController, node: PipelineController.Node) -> bool:
    """Return True to re-run the loop body (train → evaluate)."""
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
        f"Loop condition: accuracy={acc:.2f}% {'< 95% → retrain' if should_loop else '>= 95% → done'} "
        f"(iteration {node._loop_iteration})"
    )
    return should_loop


# ── Pipeline definition ─────────────────────────────────────────────

def build_pipeline(run_remote: bool = False) -> PipelineController:
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

    # The loop controller: after evaluate completes, check accuracy.
    # If < 95%, reset train+evaluate and re-run them (up to 5 times).
    pipe.add_function_step(
        name="loop_ctrl",
        function=lambda: "loop_placeholder",
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
