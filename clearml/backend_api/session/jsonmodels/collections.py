from typing import Any


class ModelCollection(list):
    """`ModelCollection` is list which validates stored values.

    Validation is made with use of field passed to `__init__` at each point,
    when new value is assigned.

    """

    def __init__(self, field: Any) -> None:
        self.field = field

    def append(self, value: Any) -> None:
        self.field.validate_single_value(value)
        super(ModelCollection, self).append(value)

    def __setitem__(self, key: int, value: Any) -> None:
        self.field.validate_single_value(value)
        super(ModelCollection, self).__setitem__(key, value)
