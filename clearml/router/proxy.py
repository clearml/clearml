from typing import Callable, Dict, Optional, Any
from fastapi import Request, Response

from .fastapi_proxy import FastAPIProxy
from .route import Route


class HttpProxy:
    DEFAULT_PORT = 9000

    def __init__(
        self,
        port: Optional[int] = None,
        workers: Optional[int] = None,
        default_target: Optional[str] = None,
        log_level: Optional[str] = None,
        access_log: bool = True,
        enable_streaming: bool = True,
    ) -> None:
        # at the moment, only a fastapi proxy is supported
        self.base_proxy = FastAPIProxy(
            port or self.DEFAULT_PORT,
            workers=workers,
            default_target=default_target,
            log_level=log_level,
            access_log=access_log,
            enable_streaming=enable_streaming,
        )
        self.base_proxy.start()
        self.port = port
        self.routes = {}

    def add_route(
        self,
        source: str,
        target: str,
        request_callback: Optional[Callable[[Request], Request]] = None,
        response_callback: Optional[Callable[[Response], Response]] = None,
        endpoint_telemetry: bool = True,
        error_callback: Optional[Callable[[Exception], Response]] = None,
    ) -> Route:
        self.routes[source] = self.base_proxy.add_route(
            source=source,
            target=target,
            request_callback=request_callback,
            response_callback=response_callback,
            endpoint_telemetry=endpoint_telemetry,
            error_callback=error_callback,
        )
        return self.routes[source]

    def remove_route(self, source: str) -> None:
        self.routes.pop(source, None)
        self.base_proxy.remove_route(source)

    def get_routes(self) -> Dict[str, Any]:
        return self.routes

    def start(self) -> None:
        self.base_proxy.start()

    def stop(self) -> None:
        self.base_proxy.stop()
