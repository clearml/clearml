import attr
from attr.validators import and_, instance_of, optional
from six import string_types
from typing import Any

# noinspection PyTypeChecker
sequence = instance_of((list, tuple))


def sequence_of(types: type) -> attr.validators._AndValidator:
    def validator(_: Any, attrib: attr.Attribute, value: Any) -> None:
        assert all(isinstance(x, types) for x in value), attrib.name

    return and_(sequence, validator)


@attr.s
class Action(object):
    name = attr.ib()
    version = attr.ib()
    service = attr.ib()
    definitions_keys = attr.ib(validator=sequence)
    authorize = attr.ib(validator=instance_of(bool), default=True)
    log_data = attr.ib(validator=instance_of(bool), default=True)
    log_result_data = attr.ib(validator=instance_of(bool), default=True)
    internal = attr.ib(default=False)
    allow_roles = attr.ib(default=None, validator=optional(sequence_of(string_types)))
    request = attr.ib(validator=optional(instance_of(dict)), default=None)
    batch_request = attr.ib(validator=optional(instance_of(dict)), default=None)
    response = attr.ib(validator=optional(instance_of(dict)), default=None)
    method = attr.ib(default=None)
    description = attr.ib(
        default=None,
        validator=optional(instance_of(string_types)),
    )
