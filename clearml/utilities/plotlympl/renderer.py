"""
Renderer Module

This module defines the PlotlyRenderer class and a single function,
fig_to_plotly, which is intended to be the main way that user's will interact
with the matplotlylib package.

"""
from __future__ import absolute_import

import warnings
from typing import Optional, TextIO, List, Union, Dict, Any

import six

from . import mpltools
from .mplexporter import Renderer


# Warning format
def warning_on_one_line(
    msg: str,
    category: type,
    filename: str,
    lineno: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None,
) -> str:
    return "%s:%s: %s:\n\n%s\n\n" % (filename, lineno, category.__name__, msg)


warnings.formatwarning = warning_on_one_line


class PlotlyRenderer(Renderer):
    """A renderer class inheriting from base for rendering mpl plots in plotly.

    A renderer class to be used with an exporter for rendering matplotlib
    plots in Plotly. This module defines the PlotlyRenderer class which handles
    the creation of the JSON structures that get sent to plotly.

    All class attributes available are defined in __init__().

    Basic Usage:

    # (mpl code) #
    fig = gcf()
    renderer = PlotlyRenderer(fig)
    exporter = Exporter(renderer)
    exporter.run(fig)  # ... et voila

    """

    def __init__(self) -> None:
        """Initialize PlotlyRenderer obj.

        PlotlyRenderer obj is called on by an Exporter object to draw
        matplotlib objects like figures, axes, text, etc.

        All class attributes are listed here in the __init__ method.

        """
        self.plotly_fig = dict(data=[], layout={})
        self.mpl_fig = None
        self.current_mpl_ax = None
        self.bar_containers = None
        self.current_bars = []
        self.current_bars_names = []
        self.axis_ct = 0
        self.x_is_mpl_date = False
        self.mpl_x_bounds = (0, 1)
        self.mpl_y_bounds = (0, 1)
        self.msg = "Initialized PlotlyRenderer\n"

    def open_figure(self, fig: Any, props: dict) -> None:
        """Creates a new figure by beginning to fill out layout dict.

        The 'autosize' key is set to false so that the figure will mirror
        sizes set by mpl. The 'hovermode' key controls what shows up when you
        mouse around a figure in plotly, it's set to show the 'closest' point.

        Positional agurments:
        fig -- a matplotlib.figure.Figure object.
        props.keys(): [
            'figwidth',
            'figheight',
            'dpi'
            ]

        """
        self.msg += "Opening figure\n"
        self.mpl_fig = fig
        self.plotly_fig["layout"] = dict(
            width=int(props["figwidth"] * props["dpi"]),
            height=int(props["figheight"] * props["dpi"]),
            autosize=False,
            hovermode="closest",
        )
        self.mpl_x_bounds, self.mpl_y_bounds = mpltools.get_axes_bounds(fig)
        margin = dict(
            l=int(self.mpl_x_bounds[0] * self.plotly_fig["layout"]["width"]),
            r=int((1 - self.mpl_x_bounds[1]) * self.plotly_fig["layout"]["width"]),
            t=int((1 - self.mpl_y_bounds[1]) * self.plotly_fig["layout"]["height"]),
            b=int(self.mpl_y_bounds[0] * self.plotly_fig["layout"]["height"]),
            pad=0,
        )
        self.plotly_fig["layout"]["margin"] = margin

    def close_figure(self, fig: Any) -> None:
        """Closes figure by cleaning up data and layout dictionaries.

        The PlotlyRenderer's job is to create an appropriate set of data and
        layout dictionaries. When the figure is closed, some cleanup and
        repair is necessary. This method removes inappropriate dictionary
        entries, freeing up Plotly to use defaults and best judgements to
        complete the entries. This method is called by an Exporter object.

        Positional arguments:
        fig -- a matplotlib.figure.Figure object.

        """
        self.plotly_fig["layout"]["showlegend"] = False
        self.msg += "Closing figure\n"

    def open_axes(self, ax: Any, props: dict) -> None:
        """Setup a new axes object (subplot in plotly).

        Plotly stores information about subplots in different 'xaxis' and
        'yaxis' objects which are numbered. These are just dictionaries
        included in the layout dictionary. This function takes information
        from the Exporter, fills in appropriate dictionary entries,
        and updates the layout dictionary. PlotlyRenderer keeps track of the
        number of plots by incrementing the axis_ct attribute.

        Setting the proper plot domain in plotly is a bit tricky. Refer to
        the documentation for mpltools.convert_x_domain and
        mpltools.convert_y_domain.

        Positional arguments:
        ax -- an mpl axes object. This will become a subplot in plotly.
        props.keys() -- [
            'axesbg',           (background color for axes obj)
            'axesbgalpha',      (alpha, or opacity for background)
            'bounds',           ((x0, y0, width, height) for axes)
            'dynamic',          (zoom/pan-able?)
            'axes',             (list: [xaxis, yaxis])
            'xscale',           (log, linear, or date)
            'yscale',
            'xlim',             (range limits for x)
            'ylim',
            'xdomain'           (xdomain=xlim, unless it's a date)
            'ydomain'
            ]

        """
        self.msg += "  Opening axes\n"
        self.current_mpl_ax = ax
        self.bar_containers = [c for c in ax.containers if c.__class__.__name__ == "BarContainer"]  # empty is OK
        self.current_bars = []

        # set defaults in axes
        xaxis = dict(anchor="y{0}".format(self.axis_ct or ""), zeroline=False, ticks="inside")
        yaxis = dict(anchor="x{0}".format(self.axis_ct or ""), zeroline=False, ticks="inside")
        zaxis = dict(anchor="x{0}".format(self.axis_ct or ""), zeroline=False, ticks="inside")
        # update defaults with things set in mpl
        mpl_xaxis, mpl_yaxis, mpl_zaxis = mpltools.prep_xyz_axis(
            ax=ax, props=props, x_bounds=self.mpl_x_bounds, y_bounds=self.mpl_y_bounds
        )
        xaxis.update(mpl_xaxis)
        yaxis.update(mpl_yaxis)
        zaxis.update(mpl_zaxis)
        bottom_spine = mpltools.get_spine_visible(ax, "bottom")
        top_spine = mpltools.get_spine_visible(ax, "top")
        left_spine = mpltools.get_spine_visible(ax, "left")
        right_spine = mpltools.get_spine_visible(ax, "right")
        xaxis["mirror"] = mpltools.get_axis_mirror(bottom_spine, top_spine)
        yaxis["mirror"] = mpltools.get_axis_mirror(left_spine, right_spine)
        xaxis["showline"] = bottom_spine
        yaxis["showline"] = top_spine

        # put axes in our figure
        self.plotly_fig["layout"]["xaxis{0}".format(self.axis_ct or "")] = xaxis
        self.plotly_fig["layout"]["yaxis{0}".format(self.axis_ct or "")] = yaxis
        if mpl_zaxis:
            self.plotly_fig["layout"]["zaxis{0}".format(self.axis_ct or "")] = zaxis

        # let all subsequent dates be handled properly if required

        if xaxis.get("type") == "date":
            self.x_is_mpl_date = True

        self.axis_ct += 1

    def close_axes(self, ax: Any) -> None:
        """Close the axes object and clean up.

        Bars from bar charts are given to PlotlyRenderer one-by-one,
        thus they need to be taken care of at the close of each axes object.
        The self.current_bars variable should be empty unless a bar
        chart has been created.

        Positional arguments:
        ax -- an mpl axes object, not required at this time.

        """
        if self.current_bars:
            # noinspection PyBroadException
            try:
                self.current_bars_names = [n.get_text() for n in ax.legend().texts]
            except Exception:
                pass
        self.draw_bars(self.current_bars)
        self.msg += "  Closing axes\n"
        self.x_is_mpl_date = False

    def draw_bars(self, bars: List[Dict[str, Any]]) -> None:
        # sort bars according to bar containers
        mpl_traces = []
        for container in self.bar_containers:
            mpl_traces.append([bar_props for bar_props in self.current_bars if bar_props["mplobj"] in container])
        for i, trace in enumerate(mpl_traces):
            self.draw_bar(
                trace,
                self.current_bars_names[i] if i < len(self.current_bars_names) else None,
            )

    def draw_bar(
        self,
        coll: List[Dict[str, Union[float, str, int]]],
        name: Optional[str] = None,
    ) -> None:
        """Draw a collection of similar patches as a bar chart.

        After bars are sorted, an appropriate data dictionary must be created
        to tell plotly about this data. Just like draw_line or draw_markers,
        draw_bar translates patch/path information into something plotly
        understands.

        Positional arguments:
        patch_coll -- a collection of patches to be drawn as a bar chart.

        """
        tol = 1e-10
        trace = [mpltools.make_bar(**bar_props) for bar_props in coll]
        widths = [bar_props["x1"] - bar_props["x0"] for bar_props in trace]
        heights = [bar_props["y1"] - bar_props["y0"] for bar_props in trace]
        vertical = abs(sum(widths[0] - widths[iii] for iii in range(len(widths)))) < tol
        horizontal = abs(sum(heights[0] - heights[iii] for iii in range(len(heights)))) < tol
        if (vertical and horizontal) or (not vertical and not horizontal):
            # Check for monotonic x. Can't both be true!
            x_zeros = [bar_props["x0"] for bar_props in trace]
            if all((x_zeros[iii + 1] > x_zeros[iii] for iii in range(len(x_zeros[:-1])))):
                orientation = "v"
            else:
                orientation = "h"
        elif vertical:
            orientation = "v"
        else:
            orientation = "h"
        if orientation == "v":
            self.msg += "    Attempting to draw a vertical bar chart\n"
            old_heights = [bar_props["y1"] for bar_props in trace]
            for bar in trace:
                bar["y0"], bar["y1"] = 0, bar["y1"] - bar["y0"]
            new_heights = [bar_props["y1"] for bar_props in trace]
            # check if we're stacked or not...
            for old, new in zip(old_heights, new_heights):
                if abs(old - new) > tol:
                    self.plotly_fig["layout"]["barmode"] = "stack"
                    self.plotly_fig["layout"]["hovermode"] = "x"
            x = [bar["x0"] + (bar["x1"] - bar["x0"]) / 2 for bar in trace]
            y = [bar["y1"] for bar in trace]
            bar_gap = mpltools.get_bar_gap([bar["x0"] for bar in trace], [bar["x1"] for bar in trace])
            if self.x_is_mpl_date:
                x = [bar["x0"] for bar in trace]
                formatter = self.current_mpl_ax.get_xaxis().get_major_formatter().__class__.__name__
                x = mpltools.mpl_dates_to_datestrings(x, formatter)
        else:
            self.msg += "    Attempting to draw a horizontal bar chart\n"
            old_rights = [bar_props["x1"] for bar_props in trace]
            for bar in trace:
                bar["x0"], bar["x1"] = 0, bar["x1"] - bar["x0"]
            new_rights = [bar_props["x1"] for bar_props in trace]
            # check if we're stacked or not...
            for old, new in zip(old_rights, new_rights):
                if abs(old - new) > tol:
                    self.plotly_fig["layout"]["barmode"] = "stack"
                    self.plotly_fig["layout"]["hovermode"] = "y"
            x = [bar["x1"] for bar in trace]
            y = [bar["y0"] + (bar["y1"] - bar["y0"]) / 2 for bar in trace]
            bar_gap = mpltools.get_bar_gap([bar["y0"] for bar in trace], [bar["y1"] for bar in trace])
        bar = dict(
            type="bar",
            orientation=orientation,
            x=x,
            y=y,
            xaxis="x{0}".format(self.axis_ct),
            yaxis="y{0}".format(self.axis_ct),
            opacity=trace[0]["alpha"],  # TODO: get all alphas if array?
            marker=dict(
                color=trace[0]["facecolor"],  # TODO: get all
                line=dict(width=trace[0]["edgewidth"]),
            ),
        )  # TODO ditto
        if name:
            bar["name"] = name
        if len(bar["x"]) >= 1:
            self.msg += "    Heck yeah, I drew that bar chart\n"
            self.plotly_fig["data"].append(bar)
            if bar_gap is not None:
                self.plotly_fig["layout"]["bargap"] = bar_gap
        else:
            self.msg += "    Bar chart not drawn\n"
            warnings.warn("found box chart data with length < 1, assuming data redundancy, not plotting.")

    def draw_marked_line(self, **props: Any) -> None:
        """Create a data dict for a line obj.

        This will draw 'lines', 'markers', or 'lines+markers'.

        props.keys() -- [
        'coordinates',  ('data', 'axes', 'figure', or 'display')
        'data',         (a list of xy pairs)
        'mplobj',       (the matplotlib.lines.Line2D obj being rendered)
        'label',        (the name of the Line2D obj being rendered)
        'linestyle',    (linestyle dict, can be None, see below)
        'markerstyle',  (markerstyle dict, can be None, see below)
        ]

        props['linestyle'].keys() -- [
        'alpha',        (opacity of Line2D obj)
        'color',        (color of the line if it exists, not the marker)
        'linewidth',
        'dasharray',    (code for linestyle, see DASH_MAP in mpltools.py)
        'zorder',       (viewing precedence when stacked with other objects)
        ]

        props['markerstyle'].keys() -- [
        'alpha',        (opacity of Line2D obj)
        'marker',       (the mpl marker symbol, see SYMBOL_MAP in mpltools.py)
        'facecolor',    (color of the marker face)
        'edgecolor',    (color of the marker edge)
        'edgewidth',    (width of marker edge)
        'markerpath',   (an SVG path for drawing the specified marker)
        'zorder',       (viewing precedence when stacked with other objects)
        ]

        """
        self.msg += "    Attempting to draw a line "
        line, marker = {}, {}
        if props["linestyle"] and props["markerstyle"]:
            self.msg += "... with both lines+markers\n"
            mode = "lines+markers"
        elif props["linestyle"]:
            self.msg += "... with just lines\n"
            mode = "lines"
        elif props["markerstyle"]:
            self.msg += "... with just markers\n"
            mode = "markers"
        if props["linestyle"]:
            color = mpltools.merge_color_and_opacity(props["linestyle"]["color"], props["linestyle"]["alpha"])

            # print(mpltools.convert_dash(props['linestyle']['dasharray']))
            line = dict(
                color=color,
                width=props["linestyle"]["linewidth"],
                dash=mpltools.convert_dash(props["linestyle"]["dasharray"]),
            )
        if props["markerstyle"]:
            marker = dict(
                opacity=props["markerstyle"]["alpha"],
                color=props["markerstyle"]["facecolor"],
                symbol=mpltools.convert_symbol(props["markerstyle"]["marker"]),
                size=props["markerstyle"]["markersize"],
                line=dict(
                    color=props["markerstyle"]["edgecolor"],
                    width=props["markerstyle"]["edgewidth"],
                ),
            )
        if props["coordinates"] == "data":
            marked_line = dict(
                type="scatter",
                mode=mode,
                name=(str(props["label"]) if isinstance(props["label"], six.string_types) else props["label"]),
                x=props["data"][0]
                if props.get("type") == "collection" and props.get("is_3d")
                else [xy_pair[0] for xy_pair in props["data"]],
                y=props["data"][1]
                if props.get("type") == "collection" and props.get("is_3d")
                else [xy_pair[1] for xy_pair in props["data"]],
                xaxis="x{0}".format(self.axis_ct),
                yaxis="y{0}".format(self.axis_ct),
                line=line,
                marker=marker,
            )
            if props.get("is_3d"):
                marked_line["z"] = (
                    props["data"][2]
                    if props.get("type") == "collection"
                    else [xyz_tuple[2] for xyz_tuple in props["data"]]
                )
                marked_line["type"] = "scatter3d"
                marked_line["zaxis"] = "z{0}".format(self.axis_ct)
            if self.x_is_mpl_date:
                formatter = self.current_mpl_ax.get_xaxis().get_major_formatter().__class__.__name__
                marked_line["x"] = mpltools.mpl_dates_to_datestrings(marked_line["x"], formatter)
            self.plotly_fig["data"].append(marked_line)
            self.msg += "    Heck yeah, I drew that line\n"
        else:
            self.msg += "    Line didn't have 'data' coordinates, not drawing\n"
            warnings.warn(
                "Bummer! Plotly can currently only draw Line2D "
                "objects from matplotlib that are in 'data' "
                "coordinates!"
            )

    def draw_image(self, **props: Any) -> None:
        """Draw image.

        Not implemented yet!

        """
        self.msg += "    Attempting to draw image\n"
        self.msg += "    Not drawing image\n"
        warnings.warn(
            "Aw. Snap! You're gonna have to hold off on "
            "the selfies for now. Plotly can't import "
            "images from matplotlib yet!"
        )

    def draw_path_collection(self, ax: Any, **props: Any) -> None:
        """Add a path collection to data list as a scatter plot.

        Current implementation defaults such collections as scatter plots.
        Matplotlib supports collections that have many of the same parameters
        in common like color, size, path, etc. However, they needn't all be
        the same. Plotly does not currently support such functionality and
        therefore, the style for the first object is taken and used to define
        the remaining paths in the collection.

        props.keys() -- [
        'paths',                (structure: [vertices, path_code])
        'path_coordinates',     ('data', 'axes', 'figure', or 'display')
        'path_transforms',      (mpl transform, including Affine2D matrix)
        'offsets',              (offset from axes, helpful if in 'data')
        'offset_coordinates',   ('data', 'axes', 'figure', or 'display')
        'offset_order',
        'styles',               (style dict, see below)
        'mplobj'                (the collection obj being drawn)
        ]

        props['styles'].keys() -- [
        'linewidth',            (one or more linewidths)
        'facecolor',            (one or more facecolors for path)
        'edgecolor',            (one or more edgecolors for path)
        'alpha',                (one or more opacites for path)
        'zorder',               (precedence when stacked)
        ]

        """
        self.msg += "    Attempting to draw a path collection\n"
        if props["offset_coordinates"] == "data":
            markerstyle = mpltools.get_markerstyle_from_collection(props)
            scatter_props = {
                "coordinates": "data",
                "data": props["offsets"],
                "label": None,
                "markerstyle": markerstyle,
                "linestyle": None,
                "type": "collection",
                "is_3d": "3d" in str(type(ax)).split(".")[-1].lower(),
            }
            self.msg += "    Drawing path collection as markers\n"
            self.draw_marked_line(**scatter_props)
        else:
            self.msg += "    Path collection not linked to 'data', not drawing\n"
            warnings.warn(
                "Dang! That path collection is out of this "
                "world. I totally don't know what to do with "
                "it yet! Plotly can only import path "
                "collections linked to 'data' coordinates"
            )

    def draw_path(self, **props: Any) -> None:
        """Draw path, currently only attempts to draw bar charts.

        This function attempts to sort a given path into a collection of
        horizontal or vertical bar charts. Most of the actual code takes
        place in functions from mpltools.py.

        props.keys() -- [
        'data',         (a list of verticies for the path)
        'coordinates',  ('data', 'axes', 'figure', or 'display')
        'pathcodes',    (code for the path, structure: ['M', 'L', 'Z', etc.])
        'style',        (style dict, see below)
        'mplobj'        (the mpl path object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of path obj)
        'edgecolor',
        'facecolor',
        'edgewidth',
        'dasharray',    (style for path's enclosing line)
        'zorder'        (precedence of obj when stacked)
        ]

        """
        self.msg += "    Attempting to draw a path\n"
        is_bar = mpltools.is_bar(self.current_mpl_ax.containers, **props)
        if is_bar:
            self.current_bars += [props]
        elif mpltools.is_fancy_bbox(**props):
            self.current_bars += [props]
        else:
            self.msg += "    This path isn't a bar, not drawing\n"
            warnings.warn("I found a path object that I don't think is part of a bar chart. Ignoring.")

    def draw_text(self, **props: Any) -> None:
        """Create an annotation dict for a text obj.

        Currently, plotly uses either 'page' or 'data' to reference
        annotation locations. These refer to 'display' and 'data',
        respectively for the 'coordinates' key used in the Exporter.
        Appropriate measures are taken to transform text locations to
        reference one of these two options.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "    Attempting to draw an mpl text object\n"
        if not mpltools.check_corners(props["mplobj"], self.mpl_fig):
            warnings.warn(
                "Looks like the annotation(s) you are trying \n"
                "to draw lies/lay outside the given figure size.\n\n"
                "Therefore, the resulting Plotly figure may not be \n"
                "large enough to view the full text. To adjust \n"
                "the size of the figure, use the 'width' and \n"
                "'height' keys in the Layout object. Alternatively,\n"
                "use the Margin object to adjust the figure's margins."
            )
        align = props["mplobj"]._multialignment
        if not align:
            align = props["style"]["halign"]  # mpl default
        if "annotations" not in self.plotly_fig["layout"]:
            self.plotly_fig["layout"]["annotations"] = []
        if props["text_type"] == "xlabel":
            self.msg += "      Text object is an xlabel\n"
            self.draw_xlabel(**props)
        elif props["text_type"] == "ylabel":
            self.msg += "      Text object is a ylabel\n"
            self.draw_ylabel(**props)
        elif props["text_type"] == "title":
            self.msg += "      Text object is a title\n"
            self.draw_title(**props)
        else:  # just a regular text annotation...
            self.msg += "      Text object is a normal annotation\n"
            if props["coordinates"] != "data":
                self.msg += "        Text object isn't linked to 'data' coordinates\n"
                x_px, y_px = props["mplobj"].get_transform().transform(props["position"])
                x, y = mpltools.display_to_paper(x_px, y_px, self.plotly_fig["layout"])
                xref = "paper"
                yref = "paper"
                xanchor = props["style"]["halign"]  # no difference here!
                yanchor = mpltools.convert_va(props["style"]["valign"])
            else:
                self.msg += "        Text object is linked to 'data' coordinates\n"
                x, y = props["position"]
                axis_ct = self.axis_ct
                xaxis = self.plotly_fig["layout"]["xaxis{0}".format(axis_ct or "")]
                yaxis = self.plotly_fig["layout"]["yaxis{0}".format(axis_ct or "")]
                if xaxis["range"][0] < x < xaxis["range"][1] and yaxis["range"][0] < y < yaxis["range"][1]:
                    xref = "x{0}".format(self.axis_ct)
                    yref = "y{0}".format(self.axis_ct)
                else:
                    self.msg += "            Text object is outside plotting area, making 'paper' reference.\n"
                    x_px, y_px = props["mplobj"].get_transform().transform(props["position"])
                    x, y = mpltools.display_to_paper(x_px, y_px, self.plotly_fig["layout"])
                    xref = "paper"
                    yref = "paper"
                xanchor = props["style"]["halign"]  # no difference here!
                yanchor = mpltools.convert_va(props["style"]["valign"])
            annotation = dict(
                text=(str(props["text"]) if isinstance(props["text"], six.string_types) else props["text"]),
                opacity=props["style"]["alpha"],
                x=x,
                y=y,
                xref=xref,
                yref=yref,
                align=align,
                xanchor=xanchor,
                yanchor=yanchor,
                showarrow=False,  # change this later?
                font=dict(color=props["style"]["color"], size=props["style"]["fontsize"]),
            )
            self.plotly_fig["layout"]["annotations"] += (annotation,)
            self.msg += "    Heck, yeah I drew that annotation\n"

    def draw_title(self, **props: Any) -> None:
        """Add a title to the current subplot in layout dictionary.

        If there exists more than a single plot in the figure, titles revert
        to 'page'-referenced annotations.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "        Attempting to draw a title\n"
        if len(self.mpl_fig.axes) > 1:
            self.msg += "          More than one subplot, adding title as annotation\n"
            x_px, y_px = props["mplobj"].get_transform().transform(props["position"])
            x, y = mpltools.display_to_paper(x_px, y_px, self.plotly_fig["layout"])
            annotation = dict(
                text=props["text"],
                font=dict(color=props["style"]["color"], size=props["style"]["fontsize"]),
                xref="paper",
                yref="paper",
                x=x,
                y=y,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,  # no arrow for a title!
            )
            self.plotly_fig["layout"]["annotations"] += (annotation,)
        else:
            self.msg += "          Only one subplot found, adding as a plotly title\n"
            self.plotly_fig["layout"]["title"] = props["text"]
            titlefont = dict(size=props["style"]["fontsize"], color=props["style"]["color"])
            self.plotly_fig["layout"]["titlefont"] = titlefont

    def draw_xlabel(self, **props: Any) -> None:
        """Add an xaxis label to the current subplot in layout dictionary.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "        Adding xlabel\n"
        axis_key = "xaxis{0}".format(self.axis_ct or "")
        # bugfix: add on last axis, self.axis_ct-1
        if axis_key not in self.plotly_fig["layout"]:
            axis_key = "xaxis{0}".format(max(0, self.axis_ct - 1) or "")
        self.plotly_fig["layout"][axis_key]["title"] = str(props["text"])
        titlefont = dict(size=props["style"]["fontsize"], color=props["style"]["color"])
        self.plotly_fig["layout"][axis_key]["titlefont"] = titlefont

    def draw_ylabel(self, **props: Any) -> None:
        """Add a yaxis label to the current subplot in layout dictionary.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "        Adding ylabel\n"
        axis_key = "yaxis{0}".format(self.axis_ct or "")
        # bugfix: add on last axis, self.axis_ct-1
        if axis_key not in self.plotly_fig["layout"]:
            axis_key = "yaxis{0}".format(max(0, self.axis_ct - 1) or "")
        self.plotly_fig["layout"][axis_key]["title"] = props["text"]
        titlefont = dict(size=props["style"]["fontsize"], color=props["style"]["color"])
        self.plotly_fig["layout"][axis_key]["titlefont"] = titlefont

    def resize(self) -> None:
        """Revert figure layout to allow plotly to resize.

        By default, PlotlyRenderer tries its hardest to precisely mimic an
        mpl figure. However, plotly is pretty good with aesthetics. By
        running PlotlyRenderer.resize(), layout parameters are deleted. This
        lets plotly choose them instead of mpl.

        """
        self.msg += "Resizing figure, deleting keys from layout\n"
        for key in ["width", "height", "autosize", "margin"]:
            try:
                del self.plotly_fig["layout"][key]
            except (KeyError, AttributeError):
                pass

    def strip_style(self) -> None:
        self.msg += "Stripping mpl style is no longer supported\n"
