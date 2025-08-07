import base64
import os
from os.path import expandvars, expanduser
from pathlib2 import Path
from typing import List, TYPE_CHECKING, ValuesView, Dict, Any

from ..utilities.pyhocon import HOCONConverter, ConfigTree

if TYPE_CHECKING:
    from .config import Config


def get_items(cls) -> Dict[str, Any]:
    """get key/value items from an enum-like class (members represent enumeration key/value)"""
    return {k: v for k, v in vars(cls).items() if not k.startswith("_")}


def get_options(cls) -> ValuesView:
    """get options from an enum-like class (members represent enumeration key/value)"""
    return get_items(cls).values()


def apply_environment(config: "Config") -> List[str]:
    """Apply environment variables from the configuration"""
    env_vars = config.get("environment", None)
    if not env_vars:
        return []
    if isinstance(env_vars, (list, tuple)):
        env_vars = dict(env_vars)

    keys = list(filter(None, env_vars.keys()))

    for key in keys:
        os.environ[str(key)] = str(env_vars[key] or "")

    return keys


def apply_files(config: "Config") -> None:
    """Apply files from the configuration into the local file system"""
    files = config.get("files", None)
    if not files:
        return

    if isinstance(files, (list, tuple)):
        files = dict(files)

    print("Creating files from configuration")
    for key, data in files.items():
        path = data.get("path")
        fmt = data.get("format", "string")
        target_fmt = data.get("target_format", "string")
        overwrite = bool(data.get("overwrite", True))
        contents = data.get("contents")

        target = Path(expanduser(expandvars(path)))

        # noinspection PyBroadException
        try:
            if target.is_dir():
                print("Skipped [{}]: is a directory {}".format(key, target))
                continue

            if not overwrite and target.is_file():
                print("Skipped [{}]: file exists {}".format(key, target))
                continue
        except Exception as ex:
            print("Skipped [{}]: can't access {} ({})".format(key, target, ex))
            continue

        if contents:
            try:
                if fmt == "base64":
                    contents = base64.b64decode(contents)
                    if target_fmt != "bytes":
                        contents = contents.decode("utf-8")
            except Exception as ex:
                print("Skipped [{}]: failed decoding {} ({})".format(key, fmt, ex))
                continue

        # noinspection PyBroadException
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            print("Skipped [{}]: failed creating path {} ({})".format(key, target.parent, ex))
            continue

        try:
            if target_fmt == "bytes":
                try:
                    target.write_bytes(contents)
                except TypeError:
                    # simpler error so the user won't get confused
                    raise TypeError("a bytes-like object is required")
            else:
                try:
                    if target_fmt == "json":
                        text = HOCONConverter.to_json(contents)
                    elif target_fmt in ("yaml", "yml"):
                        text = HOCONConverter.to_yaml(contents)
                    else:
                        if isinstance(contents, ConfigTree):
                            contents = contents.as_plain_ordered_dict()
                        text = str(contents)
                except Exception as ex:
                    print("Skipped [{}]: failed encoding to {} ({})".format(key, target_fmt, ex))
                    continue
                target.write_text(text)
            print("Saved [{}]: {}".format(key, target))
        except Exception as ex:
            print("Skipped [{}]: failed saving file {} ({})".format(key, target, ex))
            continue
