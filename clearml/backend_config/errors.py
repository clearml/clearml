from typing import Optional, Any


class ConfigurationError(Exception):
    """Exception class to be raised when a configuration file is invalid."""

    def __init__(self, msg: str, file_path: Optional[str] = None, *args: Any) -> None:
        super(ConfigurationError, self).__init__(msg, *args)
        self.file_path = file_path
