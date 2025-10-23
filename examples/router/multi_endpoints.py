"""
Minimal example showing how to request multiple external endpoints for a Task.

This script assumes you already have local services bound to ports 8000 and 8001.
You can reuse `simple_webserver.py` for the primary endpoint and launch any other
service on the secondary port before running this example.
"""

from clearml import Task


if __name__ == "__main__":
    task = Task.init(project_name="Router Example", task_name="Multiple Endpoints Example")

    # Request the legacy HTTP endpoint (no suffix) and wait until it is ready.
    task.request_external_endpoint(port=8000, protocol="http", wait=True)

    # Request another HTTP endpoint on the same Task, giving it a custom name.
    task.request_external_endpoint(port=8001, protocol="http", endpoint_name="api", wait=True)

    # Optionally list all registered endpoints for quick inspection.
    for endpoint in task.list_external_endpoints():
        print(endpoint)
