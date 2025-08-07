# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import difflib
import os
import re
import sys
from typing import Callable, Union, List, Tuple, Any, Generator
from typing import Dict as _Dict

PY32 = sys.version_info[:2] == (3, 2)

if sys.version_info[0] == 3:
    binary_type = bytes
else:
    binary_type = str


class Dict(dict):
    """Convert dict key object to attribute."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Dict, self).__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError('"{0}"'.format(name))

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def parse_reqs(fpath: str) -> _Dict[str, str]:
    pkg_v_re = re.compile(r"^(?P<pkg>[^><==]+)[><==]{,2}(?P<version>.*)$")
    """Parse requirements file."""
    reqs = dict()
    with open(fpath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = pkg_v_re.match(line.strip())
            if m:
                d = m.groupdict()
                reqs[d["pkg"].strip()] = d["version"].strip()
    return reqs


def cmp_to_key(cmp_func: Callable[[Any, Any], int]) -> Any:
    """Convert a cmp=fcuntion into a key=function."""

    class K(object):
        def __init__(self, obj: Any, *args: Any) -> None:
            self.obj = obj

        def __lt__(self, other: Any) -> bool:
            return cmp_func(self.obj, other.obj) < 0

        def __gt__(self, other: Any) -> bool:
            return cmp_func(self.obj, other.obj) > 0

        def __eq__(self, other: Any) -> bool:
            return cmp_func(self.obj, other.obj) == 0

    return K


def compare_version(version1: str, version2: str) -> int:
    """Compare version number, such as 1.1.1 and 1.1b2.0."""
    v1, v2 = list(), list()

    for item in version1.split("."):
        if item.isdigit():
            v1.append(int(item))
        else:
            v1.extend([i for i in _group_alnum(item)])
    for item in version2.split("."):
        if item.isdigit():
            v2.append(int(item))
        else:
            v2.extend([i for i in _group_alnum(item)])

    while v1 and v2:
        item1, item2 = v1.pop(0), v2.pop(0)
        if item1 > item2:
            return 1
        elif item1 < item2:
            return -1

    if v1:
        return 1
    elif v2:
        return -1
    return 0


def _group_alnum(s: str) -> Generator[Union[int, str], None, None]:
    tmp = list()
    flag = 1 if s[0].isdigit() else 0
    for c in s:
        if c.isdigit():
            if flag == 0:
                yield "".join(tmp)
                tmp = list()
                flag = 1
            tmp.append(c)
        elif c.isalpha():
            if flag == 1:
                yield int("".join(tmp))
                tmp = list()
                flag = 0
            tmp.append(c)
    last = "".join(tmp)
    yield (int(last) if flag else last)


def parse_git_config(path: str) -> dict:
    """Parse git config file."""
    config = dict()
    section = None

    with open(os.path.join(path, "config"), "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                section = line[1:-1].strip()
                config[section] = dict()
            elif section:
                key, value = line.replace(" ", "").split("=")
                config[section][key] = value
    return config


def lines_diff(lines1: List[str], lines2: List[str]) -> Tuple[bool, List[str]]:
    """Show difference between lines."""
    is_diff = False
    diffs = list()

    for line in difflib.ndiff(lines1, lines2):
        if not is_diff and line[0] in ("+", "-"):
            is_diff = True
        diffs.append(line)

    return is_diff, diffs
