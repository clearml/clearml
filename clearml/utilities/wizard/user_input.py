from typing import Optional, List, Tuple


def get_input(
    key: str,
    description: str = "",
    question: str = "Enter",
    required: bool = False,
    default: Optional[str] = None,
    new_line: bool = False,
) -> Optional[str]:
    if new_line:
        print()
    while True:
        value = input("{} {} {}: ".format(question, key, description))
        if not value.strip() and required:
            print("{} is required".format(key))
        elif not (value.strip() or required):
            return default
        else:
            return value


def input_int(
    key: str,
    description: str = "",
    required: bool = False,
    default: Optional[int] = None,
    new_line: bool = False,
) -> Optional[int]:
    while True:
        try:
            value = int(
                get_input(
                    key,
                    description,
                    required=required,
                    default=default,
                    new_line=new_line,
                )
            )
            return value
        except ValueError:
            print("Invalid input: {} should be a number. Please enter an integer".format(key))


def input_bool(question: str, default: bool = False) -> bool:
    """
    :param question: string to display
    :param default: default boolean value
    :return: return True if response is 'y'/'yes' 't'/'true' in input.lower()
    """
    while True:
        try:
            response = input("{}: ".format(question)).lower()
            if not response:
                return default
            if response.startswith("y") or response.startswith("t"):
                return True
            if response.startswith("n") or response.startswith("f"):
                return False
            raise ValueError()
        except ValueError:
            print("Invalid input: please enter 'yes' or 'no'")


def input_list(
    key: str,
    description: str = "",
    question: str = "Enter",
    required: bool = False,
    default: Optional[str] = None,
    new_line: bool = False,
) -> Optional[List[str]]:
    res = get_input(key, description, question, required, default, new_line)
    if not res:
        return None

    res_list = [res]
    while input_bool("\nDefine another {}? [y/N]".format(key)):
        response = get_input(
            key=key,
            description=description,
            question=question,
            required=False,
            default=default,
            new_line=new_line,
        )
        if response:
            res_list.append(response)
    return res_list


def multiline_input(description: str = "") -> Tuple[str, int]:
    print("{} \nNote: two consecutive empty lines would terminate the input : ".format(description))
    lines = []
    empty_lines = 0
    while empty_lines < 2:
        line = input()
        lines.append(line)
        empty_lines = 0 if line else empty_lines + 1
    res = "\n".join(lines[:-1])
    return res, len(res.splitlines())
