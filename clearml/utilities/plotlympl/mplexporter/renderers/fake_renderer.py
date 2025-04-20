from typing import Optional, Any

import numpy

from .base import Renderer


class FakeRenderer(Renderer):
    """
    Fake Renderer

    This is a fake renderer which simply outputs a text tree representing the
    elements found in the plot(s).  This is used in the unit tests for the
    package.

    Below are the methods your renderer must implement. You are free to do
    anything you wish within the renderer (i.e. build an XML or JSON
    representation, call an external API, etc.)  Here the renderer just
    builds a simple string representation for testing purposes.
    """

    def __init__(self) -> None:
        self.output = ""

    def open_figure(self, fig: Any, props: Any) -> None:
        self.output += "opening figure\n"

    def close_figure(self, fig: Any) -> None:
        self.output += "closing figure\n"

    def open_axes(self, ax: Any, props: Any) -> None:
        self.output += "  opening axes\n"

    def close_axes(self, ax: Any) -> None:
        self.output += "  closing axes\n"

    def open_legend(self, legend: Any, props: Any) -> None:
        self.output += "    opening legend\n"

    def close_legend(self, legend: Any) -> None:
        self.output += "    closing legend\n"

    def draw_text(
        self,
        text: str,
        position: Any,
        coordinates: Any,
        style: Any,
        text_type: Optional[str] = None,
        mplobj: Optional[Any] = None,
    ) -> None:
        self.output += "    draw text '{0}' {1}\n".format(text, text_type)

    def draw_path(
        self,
        data: numpy.ndarray,
        coordinates: str,
        pathcodes: numpy.ndarray,
        style: str,
        offset: Optional[numpy.ndarray] = None,
        offset_coordinates: str = "data",
        mplobj: Optional[Any] = None,
    ) -> None:
        self.output += "    draw path with {0} vertices\n".format(data.shape[0])

    def draw_image(
        self,
        imdata: Any,
        extent: Any,
        coordinates: Any,
        style: Any,
        mplobj: Optional[Any] = None,
    ) -> None:
        self.output += "    draw image of size {0}\n".format(len(imdata))


class FullFakeRenderer(FakeRenderer):
    """
    Renderer with the full complement of methods.

    When the following are left undefined, they will be implemented via
    other methods in the class.  They can be defined explicitly for
    more efficient or specialized use within the renderer implementation.
    """

    def draw_line(
        self,
        data: numpy.ndarray,
        coordinates: str,
        style: str,
        label: str,
        mplobj: Optional[Any] = None,
    ) -> None:
        self.output += "    draw line with {0} points\n".format(data.shape[0])

    def draw_markers(
        self,
        data: numpy.ndarray,
        coordinates: str,
        style: str,
        label: str,
        mplobj: Optional[Any] = None,
    ) -> None:
        self.output += "    draw {0} markers\n".format(data.shape[0])

    def draw_path_collection(
        self,
        paths: Any,
        path_coordinates: Any,
        path_transforms: Any,
        offsets: Any,
        offset_coordinates: str,
        offset_order: Any,
        styles: Any,
        mplobj: Optional[Any] = None,
    ) -> None:
        self.output += "    draw path collection with {0} offsets\n".format(offsets.shape[0])
