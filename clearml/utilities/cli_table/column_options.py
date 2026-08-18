class CliTableColumnOptions:
    """
    Layout options of a single column of a command-line table.

    :param int min_width: Width the column takes when its entries are shorter; the table widens the
        column to its longest entry, header included, whenever that entry is wider.
    :param str format_options: Prefix of the Python format specification applied to the column's
        body entries, holding the fill and alignment (e.g. ">", "^", "*<"); the width is appended
        to it. Column headers are laid out to the width alone, so alignment never applies to them.
    """

    def __init__(
        self,
        min_width: int,
        format_options: str = "<",
    ):
        self.min_width = min_width
        self.format_options = format_options
