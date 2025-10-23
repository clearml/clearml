from ..api_proxy import ApiServiceProxy


# allow us to replace the api version after we have completed the authentication process,
# and we know the backend server api version.
# ApiServiceProxy will dynamically load the correct api object based on the session api_version
auth = ApiServiceProxy("auth")
events = ApiServiceProxy("events")
models = ApiServiceProxy("models")
projects = ApiServiceProxy("projects")
tasks = ApiServiceProxy("tasks")
workers = ApiServiceProxy("workers")
queues = ApiServiceProxy("queues")
datasets = ApiServiceProxy("datasets")
dataviews = ApiServiceProxy("dataviews")
frames = ApiServiceProxy("frames")

__all__ = [
    "auth",
    "events",
    "models",
    "projects",
    "tasks",
    "workers",
    "queues",
    "datasets",
    "dataviews",
    "frames",
]
