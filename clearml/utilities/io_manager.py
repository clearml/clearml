from typing import Any


class IOCallsManager(object):
    def __init__(self) -> None:
        self.threads_io = {}

    def add_io_to_thread(self, thread_id: int, io_object: Any) -> None:
        if thread_id in self.threads_io:
            self.threads_io[thread_id].add(id(io_object))
        else:
            self.threads_io[thread_id] = {id(io_object)}
        if self._io_has_canvas_figure(io_object):
            self.threads_io[thread_id].add(id(io_object.canvas.figure))

    def is_plot_called(self, thread_id: int, io_object: Any) -> bool:
        return id(io_object) in self.threads_io.get(thread_id, set())

    def remove_io_to_thread(self, thread_id: int, io_object: Any) -> None:
        try:
            self.threads_io[thread_id].remove(id(io_object))
            if self._io_has_canvas_figure(io_object):
                self.threads_io[thread_id].remove(id(io_object.canvas.figure))
        except Exception:
            pass

    def remove_thread(self, thread_id: int) -> None:
        if thread_id in self.threads_io:
            del self.threads_io[thread_id]

    @staticmethod
    def _io_has_canvas_figure(io_object: Any) -> bool:
        return hasattr(io_object, "canvas") and hasattr(io_object.canvas, "figure")
