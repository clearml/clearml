import sys
import os
from logging import getLogger
from typing import Optional, Callable, Any
from urllib.parse import urlparse

from .frameworks import _patched_call  # noqa
from .import_bind import PostImportHookPatching
from ..config import running_remotely


class PatchGradio:
    _current_task = None
    __patched = False

    _default_gradio_address = "0.0.0.0"
    _default_gradio_port = 7860
    _root_path_format = "/service/{}/"
    _gradio_static_route_env_var = "CLEARML_GRADIO_STATIC_ROUTE"
    _gradio_endpoint = None
    _request_external_endpoint_retries = 3
    __server_config_warning = set()

    @classmethod
    def _get_root_path(cls):
        # if there is no endpoint -> request one the moment root path is required
        if not cls._gradio_endpoint:
            for _ in range(cls._request_external_endpoint_retries):
                cls._gradio_endpoint = cls._current_task.request_external_endpoint(
                    port=cls._default_gradio_port,
                    static_route=os.environ.get(cls._gradio_static_route_env_var),
                    wait=True
                )
                if cls._gradio_endpoint:
                    break
        if not cls._gradio_endpoint or not cls._gradio_endpoint.get("endpoint"):
            raise ValueError("Gradio bindings could not retrieve external endpoint")
        return urlparse(cls._gradio_endpoint.get("endpoint")).path

    @classmethod
    def update_current_task(cls, task: Optional[Any] = None) -> None:
        cls._current_task = task
        if not task:
            # Reset accumulated state when task is None (cleanup for test isolation)
            # Note: __patched is NOT reset - monkey-patching should persist
            cls._PatchGradio__server_config_warning = set()
            return
        if cls.__patched:
            return
        if "gradio" in sys.modules:
            cls.patch_gradio()
        else:
            PostImportHookPatching.add_on_import("gradio", cls.patch_gradio)

    @classmethod
    def patch_gradio(cls) -> None:
        if cls.__patched:
            return
        # noinspection PyBroadException
        try:
            import gradio

            gradio.routes.App.get_blocks = _patched_call(gradio.routes.App.get_blocks, PatchGradio._patched_get_blocks)
            gradio.blocks.Blocks.launch = _patched_call(gradio.blocks.Blocks.launch, PatchGradio._patched_launch)
        except Exception:
            pass
        cls.__patched = True

    @staticmethod
    def _patched_get_blocks(original_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        blocks = original_fn(*args, **kwargs)
        if not PatchGradio._current_task or not running_remotely():
            return blocks
        blocks.config["root"] = PatchGradio._get_root_path()
        blocks.root = blocks.config["root"]
        return blocks

    @staticmethod
    def _patched_launch(original_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not PatchGradio._current_task:
            return original_fn(*args, **kwargs)
        PatchGradio.__warn_on_server_config(
            kwargs.get("server_name"),
            kwargs.get("server_port"),
            kwargs.get("root_path"),
        )
        if not running_remotely():
            return original_fn(*args, **kwargs)

        kwargs["server_name"] = PatchGradio._default_gradio_address
        kwargs["server_port"] = PatchGradio._default_gradio_port
        kwargs["root_path"] = PatchGradio._get_root_path()
        # noinspection PyBroadException
        try:
            return original_fn(*args, **kwargs)
        except Exception:
            del kwargs["root_path"]
            return original_fn(*args, **kwargs)

    @classmethod
    def __warn_on_server_config(
        cls,
        server_name: Optional[str],
        server_port: Optional[int],
        root_path: Optional[str],
    ) -> None:
        if (server_name is None or server_name == PatchGradio._default_gradio_address) and (
            server_port is None and server_port == PatchGradio._default_gradio_port
        ):
            return
        if (server_name, server_port, root_path) in cls.__server_config_warning:
            return
        cls.__server_config_warning.add((server_name, server_port, root_path))
        if server_name is not None and server_port is not None:
            server_config = "{}:{}".format(server_name, server_port)
            what_to_ignore = "name and port"
        elif server_name is not None:
            server_config = str(server_name)
            what_to_ignore = "name"
        else:
            server_config = str(server_port)
            what_to_ignore = "port"
        getLogger().warning(
            "ClearML only supports '{}:{}' as the Gradio server. Ignoring {} '{}' in remote execution".format(
                PatchGradio._default_gradio_address,
                PatchGradio._default_gradio_port,
                what_to_ignore,
                server_config,
            )
        )
        if root_path is not None:
            getLogger().warning(
                "ClearML will override root_path '{}' to '{}' in remote execution".format(
                    root_path,
                    PatchGradio._get_root_path(),
                )
            )
