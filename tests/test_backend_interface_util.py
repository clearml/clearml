import re
from types import SimpleNamespace

from clearml.backend_interface.util import get_or_create_project


class NormalizingProjectSession:
    def __init__(self):
        self.projects = {}

    def send(self, request):
        if request.__class__.__name__ == "GetAllRequest":
            matches = [
                SimpleNamespace(id=project_id, system_tags=[])
                for name, project_id in self.projects.items()
                if re.fullmatch(request.name, name)
            ]
            return SimpleNamespace(response=SimpleNamespace(projects=matches))
        if request.__class__.__name__ == "CreateRequest":
            normalized_name = request.name.lstrip("/")
            if normalized_name in self.projects:
                raise RuntimeError("Project with the same name already exists")
            self.projects[normalized_name] = "project-id"
            return SimpleNamespace(response=SimpleNamespace(id="project-id"))
        raise AssertionError(type(request))


def test_get_or_create_project_normalizes_leading_slashes():
    session = NormalizingProjectSession()

    first_id = get_or_create_project(session, "/tmp/test/runs")
    second_id = get_or_create_project(session, "/tmp/test/runs")

    assert first_id == second_id == "project-id"
    assert session.projects == {"tmp/test/runs": "project-id"}
