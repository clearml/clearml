import sys
import types
from collections import defaultdict
from typing import Callable, Any

import six

if six.PY2:
    # python2.x
    import __builtin__ as builtins
else:
    # python3.x
    import builtins


class PostImportHookPatching(object):
    _patched = False
    _post_import_hooks = defaultdict(list)

    @staticmethod
    def _init_hook() -> None:
        if PostImportHookPatching._patched:
            return
        PostImportHookPatching._patched = True

        if six.PY2:
            # python2.x
            builtins.__org_import__ = builtins.__import__
            builtins.__import__ = PostImportHookPatching.__patched_import2
        else:
            # python3.x
            builtins.__org_import__ = builtins.__import__
            builtins.__import__ = PostImportHookPatching.__patched_import3

    @staticmethod
    def __patched_import2(
        name: str,
        globals: dict = {},
        locals: dict = {},
        fromlist: list = [],
        level: int = -1,
    ) -> types.ModuleType:
        already_imported = name in sys.modules
        mod = builtins.__org_import__(name, globals=globals, locals=locals, fromlist=fromlist, level=level)

        if not already_imported and name in PostImportHookPatching._post_import_hooks:
            for hook in PostImportHookPatching._post_import_hooks[name]:
                hook()
        return mod

    @staticmethod
    def __patched_import3(
        name: str,
        globals: dict = None,
        locals: dict = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        name_parts = name.split(".")
        base_name = name_parts[0]
        second_name = ".".join(name_parts[:2]) if len(name_parts) > 1 else None
        base_already_imported = (not base_name) or (base_name in sys.modules)
        second_already_imported = (not second_name) or (second_name in sys.modules)
        mod = builtins.__org_import__(name, globals=globals, locals=locals, fromlist=fromlist, level=level)
        if not base_already_imported and base_name in PostImportHookPatching._post_import_hooks:
            for hook in PostImportHookPatching._post_import_hooks[base_name]:
                hook()
        if not second_already_imported and second_name in PostImportHookPatching._post_import_hooks:
            for hook in PostImportHookPatching._post_import_hooks[second_name]:
                hook()
        return mod

    @staticmethod
    def add_on_import(name: str, func: Callable) -> None:
        PostImportHookPatching._init_hook()
        if (
            name not in PostImportHookPatching._post_import_hooks
            or func not in PostImportHookPatching._post_import_hooks[name]
        ):
            PostImportHookPatching._post_import_hooks[name].append(func)

    @staticmethod
    def remove_on_import(name: str, func: Callable) -> None:
        if (
            name in PostImportHookPatching._post_import_hooks
            and func in PostImportHookPatching._post_import_hooks[name]
        ):
            PostImportHookPatching._post_import_hooks[name].remove(func)
