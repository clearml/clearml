import functools
import threading
from multiprocessing import Process
from typing import Optional, Callable, Awaitable, AsyncGenerator, Union

import fastapi
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from .route import Route
from ..utilities.process.mp import SafeQueue


class FastAPIProxy:
    ALL_REST_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]

    def __init__(
        self,
        port: int,
        workers: Optional[int] = None,
        default_target: Optional[str] = None,
        log_level: Optional[str] = None,
        access_log: Optional[bool] = None,
        enable_streaming: bool = True,
    ) -> None:
        self.app = None
        self.routes = {}
        self.port = port
        self.message_queue = SafeQueue()
        self.uvicorn_subprocess = None
        self.workers = workers
        self.access_log = access_log
        self.log_level = None
        self.enable_streaming = enable_streaming
        self._default_target = default_target
        self._default_session = None
        self._in_subprocess = False

    def _create_default_route(self) -> None:
        proxy = self

        class DefaultRouteMiddleware(BaseHTTPMiddleware):
            async def dispatch(
                self,
                request: Request,
                call_next: Callable[[Request], Awaitable[Response]],
            ) -> Response:
                scope = {
                    "type": "http",
                    "method": request.method,
                    "path": request.url.path,
                    "root_path": "",
                    "headers": request.headers.raw,
                    "query_string": request.url.query.encode("utf-8"),
                    "client": request.client,
                    "server": request.scope.get("server"),
                    "scheme": request.url.scheme,
                    "extensions": request.scope.get("extensions", {}),
                    "app": request.scope.get("app"),
                }
                for route in proxy.app.router.routes:
                    if route.matches(scope)[0] == Match.FULL:
                        return await call_next(request)
                proxied_response = await proxy._send_request(
                    request,
                    proxy._default_target,
                    proxy._default_target + request.url.path,
                )
                return await proxy._convert_httpx_response_to_fastapi(proxied_response)

        self.app.add_middleware(DefaultRouteMiddleware)

    async def proxy(
        self,
        request: Request,
        path: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Response:
        route_data = self.routes.get(source_path)
        if not route_data:
            return Response(status_code=404)

        request = await route_data.on_request(request)
        try:
            proxied_response = await self._send_request(
                request,
                route_data.session,
                url=f"{route_data.target_url}/{path}" if path else route_data.target_url,
            )
            proxied_response = await self._convert_httpx_response_to_fastapi(proxied_response)
        except Exception as e:
            await route_data.on_error(request, e)
            raise
        return await route_data.on_response(proxied_response, request)

    async def _send_request(self, request: Request, session: httpx.AsyncClient, url: str) -> httpx.Response:
        if not self.enable_streaming:
            proxied_response = await session.request(
                method=request.method,
                url=url,
                headers=dict(request.headers),
                content=await request.body(),
                params=request.query_params,
            )
        else:
            request = session.build_request(
                method=request.method,
                url=url,
                content=request.stream(),
                params=request.query_params,
                headers=dict(request.headers),
                timeout=httpx.USE_CLIENT_DEFAULT,
            )
            proxied_response = await session.send(
                request=request,
                auth=httpx.USE_CLIENT_DEFAULT,
                follow_redirects=httpx.USE_CLIENT_DEFAULT,
                stream=True,
            )
        return proxied_response

    async def _convert_httpx_response_to_fastapi(self, httpx_response: httpx.Response) -> fastapi.responses.Response:
        if self.enable_streaming and httpx_response.headers.get("transfer-encoding", "").lower() == "chunked":

            async def upstream_body_generator() -> AsyncGenerator[bytes, None]:
                async for chunk in httpx_response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                upstream_body_generator(),
                status_code=httpx_response.status_code,
                headers=dict(httpx_response.headers),
            )
        if not self.enable_streaming:
            content = httpx_response.content
        else:
            content = await httpx_response.aread()
        fastapi_response = Response(
            content=content,
            status_code=httpx_response.status_code,
            media_type=httpx_response.headers.get("content-type", None),
            headers=dict(httpx_response.headers),
        )
        # should delete content-length when not present in the original response
        # relevant for:
        # https://datatracker.ietf.org/doc/html/rfc9112#body.content-length:~:text=MUST%20NOT%20send%20a%20Content%2DLength%20header
        if httpx_response.headers.get("content-length") is None:
            try:
                del fastapi_response.headers["content-length"]  # no pop available
            except Exception:
                pass
        return fastapi_response

    def add_route(
        self,
        source: str,
        target: str,
        request_callback: Optional[callable] = None,
        response_callback: Optional[callable] = None,
        error_callback: Optional[callable] = None,
        endpoint_telemetry: Union[bool, dict] = True,
    ) -> Optional[Route]:
        if not self._in_subprocess:
            self.message_queue.put(
                {
                    "method": "add_route",
                    "kwargs": {
                        "source": source,
                        "target": target,
                        "request_callback": request_callback,
                        "response_callback": response_callback,
                        "error_callback": error_callback,
                        "endpoint_telemetry": endpoint_telemetry,
                    },
                }
            )
            return
        should_add_route = False
        if source not in self.routes:
            should_add_route = True
        else:
            self.routes[source].stop_endpoint_telemetry()
        self.routes[source] = Route(
            target,
            request_callback=request_callback,
            response_callback=response_callback,
            error_callback=error_callback,
            session=httpx.AsyncClient(timeout=None),
        )
        if endpoint_telemetry is True:
            endpoint_telemetry = {}
        if endpoint_telemetry is not False:
            self.routes[source].set_endpoint_telemetry_args(**endpoint_telemetry)
        if self._in_subprocess:
            self.routes[source].start_endpoint_telemetry()
        if should_add_route:
            self.app.add_api_route(
                source,
                functools.partial(
                    self.proxy,
                    source_path=source,
                ),
                methods=self.ALL_REST_METHODS,
            )
            self.app.add_api_route(
                source.rstrip("/") + "/{path:path}",
                functools.partial(
                    self.proxy,
                    source_path=source,
                ),
                methods=self.ALL_REST_METHODS,
            )
        return self.routes[source]

    def remove_route(self, source: str) -> None:
        if not self._in_subprocess:
            self.message_queue.put({"method": "remove_route", "kwargs": {"source": source}})
            return
        route = self.routes.get(source)
        if route:
            route.stop_endpoint_telemetry()
        if source in self.routes:
            # we are not popping the key to prevent calling self.app.add_api_route multiple times
            # when self.add_route is called on the same source_path after removal
            self.routes[source] = None

    def _start(self) -> None:
        self._in_subprocess = True
        self.app = FastAPI()
        if self._default_target:
            self._default_session = httpx.AsyncClient(timeout=None)
            self._create_default_route()
        for route in self.routes.values():
            route.start_endpoint_telemetry()
        threading.Thread(target=self._rpc_manager, daemon=True).start()
        uvicorn.run(
            self.app,
            port=self.port,
            host="0.0.0.0",
            workers=self.workers,
            log_level=self.log_level,
            access_log=self.access_log,
        )

    def _rpc_manager(self) -> None:
        while True:
            message = self.message_queue.get()
            if message["method"] == "add_route":
                self.add_route(**message["kwargs"])
            elif message["method"] == "remove_route":
                self.remove_route(**message["kwargs"])

    def start(self) -> None:
        self.uvicorn_subprocess = Process(target=self._start)
        self.uvicorn_subprocess.start()

    def stop(self) -> None:
        if self.uvicorn_subprocess:
            self.uvicorn_subprocess.terminate()
            self.uvicorn_subprocess = None
