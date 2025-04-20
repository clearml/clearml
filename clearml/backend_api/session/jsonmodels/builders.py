"""Builders to generate in memory representation of model and fields tree."""

from __future__ import absolute_import

from collections import defaultdict
from typing import Any

import six

from . import errors
from .fields import NotSet


class Builder(object):
    def __init__(
        self,
        parent: "Builder" = None,
        nullable: bool = False,
        default: Any = NotSet,
    ) -> None:
        self.parent = parent
        self.types_builders = {}
        self.types_count = defaultdict(int)
        self.definitions = set()
        self.nullable = nullable
        self.default = default

    @property
    def has_default(self) -> bool:
        return self.default is not NotSet

    def register_type(self, type: type, builder: "Builder") -> None:
        if self.parent:
            return self.parent.register_type(type, builder)

        self.types_count[type] += 1
        if type not in self.types_builders:
            self.types_builders[type] = builder

    def get_builder(self, type: type) -> "Builder":
        if self.parent:
            return self.parent.get_builder(type)

        return self.types_builders[type]

    def count_type(self, type: type) -> int:
        if self.parent:
            return self.parent.count_type(type)

        return self.types_count[type]

    @staticmethod
    def maybe_build(value: "Builder") -> Any:
        return value.build() if isinstance(value, Builder) else value

    def add_definition(self, builder: "Builder") -> None:
        if self.parent:
            return self.parent.add_definition(builder)

        self.definitions.add(builder)


class ObjectBuilder(Builder):
    def __init__(self, model_type: type, *args: Any, **kwargs: Any) -> None:
        super(ObjectBuilder, self).__init__(*args, **kwargs)
        self.properties = {}
        self.required = []
        self.type = model_type

        self.register_type(self.type, self)

    def add_field(self, name: str, field: Any, schema: Any) -> None:
        _apply_validators_modifications(schema, field)
        self.properties[name] = schema
        if field.required:
            self.required.append(name)

    def build(self) -> str:
        builder = self.get_builder(self.type)
        if self.is_definition and not self.is_root:
            self.add_definition(builder)
            [self.maybe_build(value) for _, value in self.properties.items()]
            return "#/definitions/{name}".format(name=self.type_name)
        else:
            return builder.build_definition(nullable=self.nullable)

    @property
    def type_name(self) -> str:
        module_name = "{module}.{name}".format(
            module=self.type.__module__,
            name=self.type.__name__,
        )
        return module_name.replace(".", "_").lower()

    def build_definition(self, add_defintitions: bool = True, nullable: bool = False) -> dict:
        properties = dict((name, self.maybe_build(value)) for name, value in self.properties.items())
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }
        if self.required:
            schema["required"] = self.required
        if self.definitions and add_defintitions:
            schema["definitions"] = dict(
                (builder.type_name, builder.build_definition(False, False)) for builder in self.definitions
            )
        return schema

    @property
    def is_definition(self) -> bool:
        if self.count_type(self.type) > 1:
            return True
        elif self.parent:
            return self.parent.is_definition
        else:
            return False

    @property
    def is_root(self) -> bool:
        return not bool(self.parent)


def _apply_validators_modifications(field_schema: dict, field: Any) -> None:
    for validator in field.validators:
        try:
            validator.modify_schema(field_schema)
        except AttributeError:
            pass


class PrimitiveBuilder(Builder):
    def __init__(self, type: type, *args: Any, **kwargs: Any) -> None:
        super(PrimitiveBuilder, self).__init__(*args, **kwargs)
        self.type = type

    def build(self) -> dict:
        schema = {}
        if issubclass(self.type, six.string_types):
            obj_type = "string"
        elif issubclass(self.type, bool):
            obj_type = "boolean"
        elif issubclass(self.type, int):
            obj_type = "number"
        elif issubclass(self.type, float):
            obj_type = "number"
        else:
            raise errors.FieldNotSupported("Can't specify value schema!", self.type)

        if self.nullable:
            obj_type = [obj_type, "null"]
        schema["type"] = obj_type

        if self.has_default:
            schema["default"] = self.default

        return schema


class ListBuilder(Builder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ListBuilder, self).__init__(*args, **kwargs)
        self.schemas = []

    def add_type_schema(self, schema: Any) -> None:
        self.schemas.append(schema)

    def build(self) -> dict:
        schema = {"type": "array"}
        if self.nullable:
            self.add_type_schema({"type": "null"})

        if self.has_default:
            schema["default"] = [self.to_struct(i) for i in self.default]

        schemas = [self.maybe_build(s) for s in self.schemas]
        if len(schemas) == 1:
            items = schemas[0]
        else:
            items = {"oneOf": schemas}

        schema["items"] = items
        return schema

    @property
    def is_definition(self) -> bool:
        return self.parent.is_definition

    @staticmethod
    def to_struct(item: Any) -> Any:
        from .models import Base

        if isinstance(item, Base):
            return item.to_struct()
        return item


class EmbeddedBuilder(Builder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(EmbeddedBuilder, self).__init__(*args, **kwargs)
        self.schemas = []

    def add_type_schema(self, schema: Any) -> None:
        self.schemas.append(schema)

    def build(self) -> dict:
        if self.nullable:
            self.add_type_schema({"type": "null"})

        schemas = [self.maybe_build(schema) for schema in self.schemas]
        if len(schemas) == 1:
            schema = schemas[0]
        else:
            schema = {"oneOf": schemas}

        if self.has_default:
            # The default value of EmbeddedField is expected to be an instance
            # of a subclass of models.Base, thus have `to_struct`
            schema["default"] = self.default.to_struct()

        return schema

    @property
    def is_definition(self) -> bool:
        return self.parent.is_definition
