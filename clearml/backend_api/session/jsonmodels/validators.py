"""Predefined validators."""
import re
from typing import Any

from six.moves import reduce

from .errors import ValidationError
from . import utilities


class Min:
    """Validator for minimum value."""

    def __init__(self, minimum_value: Any, exclusive: bool = False) -> None:
        """Init.

        :param minimum_value: Minimum value for validator.
        :param bool exclusive: If `True`, then validated value must be strongly
            lower than given threshold.

        """
        self.minimum_value = minimum_value
        self.exclusive = exclusive

    def validate(self, value: Any) -> None:
        """Validate value."""
        if self.exclusive:
            if value <= self.minimum_value:
                raise ValidationError(f"'{value}' is lower or equal than minimum ('{self.minimum_value}').")
        else:
            if value < self.minimum_value:
                raise ValidationError(f"'{value}' is lower than minimum ('{self.minimum_value}').")

    def modify_schema(self, field_schema: dict) -> None:
        """Modify field schema."""
        field_schema["minimum"] = self.minimum_value
        if self.exclusive:
            field_schema["exclusiveMinimum"] = True


class Max:
    """Validator for maximum value."""

    def __init__(self, maximum_value: Any, exclusive: bool = False) -> None:
        """Init.

        :param maximum_value: Maximum value for validator.
        :param bool exclusive: If `True`, then validated value must be strongly
            bigger than given threshold.

        """
        self.maximum_value = maximum_value
        self.exclusive = exclusive

    def validate(self, value: Any) -> None:
        """Validate value."""
        if self.exclusive:
            if value >= self.maximum_value:
                raise ValidationError(f"'{value}' is bigger or equal than maximum ('{self.maximum_value}').")
        else:
            if value > self.maximum_value:
                raise ValidationError(f"'{value}' is bigger than maximum ('{self.maximum_value}').")

    def modify_schema(self, field_schema: dict) -> None:
        """Modify field schema."""
        field_schema["maximum"] = self.maximum_value
        if self.exclusive:
            field_schema["exclusiveMaximum"] = True


class Regex:
    """Validator for regular expressions."""

    FLAGS = {
        "ignorecase": re.I,
        "multiline": re.M,
    }

    def __init__(self, pattern: str, **flags: Any) -> None:
        """Init.

        Note, that if given pattern is ECMA regex, given flags will be
        **completely ignored** and taken from given regex.


        :param string pattern: Pattern of regex.
        :param bool flags: Flags used for the regex matching.
            Allowed flag names are in the `FLAGS` attribute. The flag value
            does not matter as long as it evaluates to True.
            Flags with False values will be ignored.
            Invalid flags will be ignored.

        """
        if utilities.is_ecma_regex(pattern):
            result = utilities.convert_ecma_regex_to_python(pattern)
            self.pattern, self.flags = result
        else:
            self.pattern = pattern
            self.flags = [self.FLAGS[key] for key, value in flags.items() if key in self.FLAGS and value]

    def validate(self, value: str) -> None:
        """Validate value."""
        flags = self._calculate_flags()

        try:
            result = re.search(self.pattern, value, flags)
        except TypeError as te:
            raise ValidationError(*te.args)

        if not result:
            raise ValidationError(f'Value "{value}" did not match pattern "{self.pattern}".')

    def _calculate_flags(self) -> int:
        return reduce(lambda x, y: x | y, self.flags, 0)

    def modify_schema(self, field_schema: dict) -> None:
        """Modify field schema."""
        field_schema["pattern"] = utilities.convert_python_regex_to_ecma(self.pattern, self.flags)


class Length:
    """Validator for length."""

    def __init__(self, minimum_value: int = None, maximum_value: int = None) -> None:
        """Init.

        Note that if no `minimum_value` neither `maximum_value` will be
        specified, `ValueError` will be raised.

        :param int minimum_value: Minimum value (optional).
        :param int maximum_value: Maximum value (optional).

        """
        if minimum_value is None and maximum_value is None:
            raise ValueError("Either 'minimum_value' or 'maximum_value' must be specified.")

        self.minimum_value = minimum_value
        self.maximum_value = maximum_value

    def validate(self, value: Any) -> None:
        """Validate value."""
        len_ = len(value)

        if self.minimum_value is not None and len_ < self.minimum_value:
            raise ValidationError(f"Value '{value}' length is lower than allowed minimum '{self.minimum_value}'.")

        if self.maximum_value is not None and len_ > self.maximum_value:
            raise ValidationError(f"Value '{value}' length is bigger than allowed maximum '{self.maximum_value}'.")

    def modify_schema(self, field_schema: dict) -> None:
        """Modify field schema."""
        if self.minimum_value:
            field_schema["minLength"] = self.minimum_value

        if self.maximum_value:
            field_schema["maxLength"] = self.maximum_value


class Enum:
    """Validator for enums."""

    def __init__(self, *choices: Any) -> None:
        """Init.

        :param [] choices: Valid choices for the field.
        """

        self.choices = list(choices)

    def validate(self, value: Any) -> None:
        if value not in self.choices:
            raise ValidationError(f"Value '{value}' is not a valid choice.")

    def modify_schema(self, field_schema: dict) -> None:
        field_schema["enum"] = self.choices
