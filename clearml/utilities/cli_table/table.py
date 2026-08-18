from typing import List, Optional

from clearml.utilities.cli_table.column_options import CliTableColumnOptions


class CliTable:
    """
    A command-line table of text entries laid out in fixed-width columns, rendered as a string.

    :param list column_options: Layout options of each column, in column order; they also fix how
        many entries a row holds. Their format options apply to body rows only, and their minimum
        widths only bound the widths the table computes from the entries when rendering.
    :param list column_headers: The header row's entries, one per column, laid out to the column
        width alone.
    :param list rows: The body rows added so far, in render order, each a list of entries.
    :param str column_separator: Text inserted between two adjacent entries of a row.
    :param str header_underline: Character repeated under the header row over the full table width.
    """

    def __init__(
        self,
        column_options: List[CliTableColumnOptions],
        column_headers: List[str],
        rows: List[List[str]],
        column_separator: str = " | ",
        header_underline: str = "-",
    ):
        self.column_options = column_options
        self.column_headers = column_headers
        self.rows = rows
        self.column_separator = column_separator
        self.header_underline = header_underline

    @staticmethod
    def validate_column_options(column_options: List[CliTableColumnOptions]) -> None:
        """
        Check that at least one column is described and that each one can lay an entry out.

        :param list column_options: Layout options of each column, in column order.
        :raises ValueError: If no column is described, if a minimum width is not a positive integer,
            or if format options do not form a valid specification once a width is appended to them.
        """
        if not column_options:
            raise ValueError("Expected at least one column, got none")

        for options in column_options:
            if (
                not isinstance(options.min_width, int)  # rfh
                or isinstance(options.min_width, bool)
                or options.min_width <= 0
            ):
                raise ValueError(
                    f"Expected a positive integer column minimum width, got {options.min_width!r}"
                )

            if not isinstance(options.format_options, str):
                raise ValueError(
                    f"Expected string column format options, got {options.format_options!r}"
                )

            try:
                format("", f"{options.format_options}{options.min_width}")
            except ValueError as error:
                raise ValueError(
                    f"Invalid column format options: {options.format_options!r}"
                ) from error

    @classmethod
    def validate_row(
        cls,
        column_options: List[CliTableColumnOptions],
        cells: List[str],
    ) -> None:
        """
        Check that a row holds exactly one entry per column.

        :param list column_options: Layout options of each column, in column order.
        :param list cells: The row's entries, one per column, in column order.
        :raises ValueError: If the row holds a different number of entries than there are columns.
        """
        if len(cells) != len(column_options):
            raise ValueError(
                f"Expected a row of {len(column_options)} entries, got {len(cells)}"
            )

    @classmethod
    def init(
        cls,
        column_options: List[CliTableColumnOptions],
        column_headers: List[str],
        rows: Optional[List[List[str]]] = None,
        column_separator: str = " | ",
        header_underline: str = "-",
    ) -> "CliTable":
        """
        Build a table from its header and the rows already at hand, to be extended with `add_row`.

        :param list column_options: Layout options of each column, in column order.
        :param list column_headers: The header row's entries, one per column.
        :param list rows: The body rows to start with, in render order; None starts an empty table.
        :param str column_separator: Text inserted between two adjacent entries of a row.
        :param str header_underline: Character repeated under the header row over the full table width.
        :returns: The table, holding the given rows.
        :rtype: CliTable
        :raises ValueError: If the column options are invalid, or if the column headers or one of
            the rows hold a different number of entries than there are columns.
        """
        cls.validate_column_options(column_options=column_options)
        if len(column_headers) != len(column_options):
            raise ValueError(
                f"Expected {len(column_options)} column headers, got {len(column_headers)}"
            )

        if rows is None:
            rows = []

        for row in rows:
            cls.validate_row(
                cells=row,
                column_options=column_options,
            )

        return cls(
            column_options=column_options,
            column_headers=column_headers,
            # Copies the list to avoid mutations on caller list
            rows=[row for row in rows],
            column_separator=column_separator,
            header_underline=header_underline,
        )

    def add_row(self, cells: List[str]) -> None:
        """
        Append a row at the bottom of the table.

        :param list cells: The row's entries, one per column, in column order.
        :raises ValueError: If the row holds a different number of entries than there are columns.
        """
        self.validate_row(
            cells=cells,
            column_options=self.column_options,
        )
        self.rows.append(cells)

    @property
    def column_widths(self) -> List[int]:
        """
        Width each column is rendered at, in column order.

        :returns: For each column, its minimum width or the length of its longest entry, header
            included, whichever is greater.
        :rtype: list
        """
        return [
            max(  # rfh
                options.min_width,
                len(self.column_headers[index]),
                *(len(row[index]) for row in self.rows),
            )
            for index, options in enumerate(self.column_options)
        ]

    def render(self) -> str:
        """
        Render the whole table, each column as wide as its longest entry or its minimum width.

        :returns: The header, its underline and every body row, joined by line breaks.
        :rtype: str
        """

        def render_row(cells: List[str], specifications: List[str]) -> str:
            """Join a row's entries, each laid out with its column's specification."""
            return self.column_separator.join(
                f"{cell:{specification}}"  # rfh
                for cell, specification in zip(cells, specifications)
            )

        header_row = render_row(
            cells=self.column_headers,
            specifications=[
                f"{width}"  # rfh
                for width in self.column_widths
            ],
        )

        return "\n".join(
            (
                header_row,
                self.header_underline * len(header_row),
                *(
                    render_row(
                        cells=row,
                        specifications=[
                            f"{options.format_options}{width}"  # rfh
                            for options, width in zip(
                                self.column_options, self.column_widths
                            )
                        ],
                    )  # rfh
                    for row in self.rows
                ),
            )
        )
