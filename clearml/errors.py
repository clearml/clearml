from typing import List, Any


class UsageError(RuntimeError):
    """An exception raised for illegal usage of clearml objects"""

    pass


class ArtifactUriDeleteError(ValueError):
    def __init__(self, artifact: str, uri: str, remaining_uris: List[str]) -> None:
        super(ArtifactUriDeleteError, self).__init__("Failed deleting artifact {}: file {}".format(artifact, uri))
        self.artifact = artifact
        self.uri = uri
        self._remaining_uris = remaining_uris

    @property
    def remaining_uris(self) -> Any:
        """Remaining URIs to delete. Deletion of these URIs was aborted due to the error."""
        return self._remaining_uris
