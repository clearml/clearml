import json
from typing import List, Optional, Dict, Union, Tuple, Any

import numpy
import numpy as np

from ..errors import UsageError
from ..utilities.dicts import merge_dicts

try:
    import pandas as pd
except ImportError:
    pd = None
from attr import attrs, attrib


def create_2d_histogram_plot(
    np_row_wise: numpy.ndarray,
    labels: list,
    title: str = None,
    xtitle: str = None,
    ytitle: str = None,
    series: str = None,
    xlabels: list = None,
    comment: str = None,
    mode: str = "group",
    data_args: dict = None,
    layout_config: dict = None,
) -> dict:
    """
    Create a 2D Plotly histogram chart from a 2D numpy array
    :param np_row_wise: 2D numpy data array
    :param labels: Histogram labels
    :param title: Chart title
    :param xtitle: X-Series title
    :param ytitle: Y-Series title
    :param xlabels: The labels of the x axis.
    :param comment: comment underneath the title
    :param mode: multiple histograms mode. valid options are: stack / group / relative. Default is 'group'.
    :param data_args: optional extra data configuration
    :param layout_config: optional extra layout configuration
    :return: Plotly chart dict.
    """
    assert mode in ("stack", "group", "relative")

    np_row_wise = np.atleast_2d(np_row_wise)
    assert len(np_row_wise.shape) == 2, "Expected a 2D numpy array"
    use_series = bool(labels) and bool(series)

    # using labels without xlabels leads to original behavior
    if labels is not None and xlabels is None:
        assert len(labels) == np_row_wise.shape[0], "Please provide a label for each data row"
    elif xlabels is None:
        fake_label = series or ""
        labels = [fake_label] * np_row_wise.shape[0]
    elif labels:
        if len(labels) == 1:
            labels = labels * np_row_wise.shape[0]
        assert len(xlabels) == np_row_wise.shape[1]
    elif not labels and xlabels:
        labels = [series or ""]

    data = [
        _np_row_to_plotly_data_item(
            np_row=np_row_wise[i, :],
            label=labels[i] if labels else None,
            xlabels=xlabels,
            data_args=data_args,
        )
        for i in range(np_row_wise.shape[0])
    ]
    return _plotly_hist_dict(
        title=series if use_series else title,
        xtitle=xtitle,
        ytitle=ytitle,
        mode=mode,
        data=data,
        comment=comment,
        layout_config=layout_config,
    )


def _to_np_array(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        value = np.array(value)

    return value


@attrs
class SeriesInfo(object):
    name = attrib(type=str)
    data = attrib(type=np.ndarray, converter=_to_np_array)
    labels = attrib(default=None)

    @data.validator
    def _validate_data(self, _: Any, value: np.ndarray) -> None:
        if value.ndim != 2:
            raise ValueError("Expected series data to be 2D numpy array")

        if value.shape[1] != 2:
            raise ValueError("Expected series data to have 2 columns")

    def __attrs_post_init__(self) -> None:
        if (self.labels is not None) and (len(self.labels) != self.data.shape[0]):
            raise ValueError("If 'labels' is provided, it must be a list or tuple, the same length as the data")


def create_line_plot(
    title: str,
    series: List[SeriesInfo],
    xtitle: str,
    ytitle: str,
    mode: str = "lines",
    reverse_xaxis: bool = False,
    comment: Optional[str] = None,
    MAX_SIZE: Optional[int] = None,
    layout_config: Optional[dict] = None,
) -> dict:
    plotly_obj = _plotly_scatter_layout_dict(
        title=title if not comment else (title + "<br><sup>" + comment + "</sup>"),
        xaxis_title=xtitle,
        yaxis_title=ytitle,
    )

    if reverse_xaxis:
        plotly_obj["layout"]["xaxis"]["autorange"] = "reversed"

    # check maximum size of data
    MAX_SIZE = MAX_SIZE or 800000
    series_sizes = [s.data.size for s in series]
    total_size = sum(series_sizes)
    if total_size > MAX_SIZE:
        # we need to downscale
        base_size = MAX_SIZE / len(series_sizes)
        baseused_size = sum([min(s, base_size) for s in series_sizes])
        leftover = MAX_SIZE - baseused_size
        for s in series:
            # if we need to down-sample, use low-pass average filter and sampling
            if s.data.size >= base_size:
                budget = base_size
                # if we have some leftover in the budget, split based on series sizes
                if leftover > 0:
                    # calculate the relative overflow  of this series compared to all the overflows
                    # then multiply it by the leftover budget
                    budget += int(leftover * s.data.size / (total_size - baseused_size))
                step = int(np.ceil(s.data.size / float(budget)))
                x = s.data[:, 0][::-step][::-1]
                y = s.data[:, 1]
                y_low_pass = np.convolve(y, np.ones(shape=(step,), dtype=y.dtype) / float(step), mode="same")
                y = y_low_pass[::-step][::-1]
                s.data = np.array([x, y], dtype=s.data.dtype).T

            # decide on number of points between mean and max
            s_max = np.max(np.abs(s.data), axis=0)
            s_max = np.maximum(s_max, s_max * 0 + 0.01)
            digits = np.maximum(np.array([1, 1]), np.array([6, 6]) - np.floor(np.abs(np.log10(s_max))))
            s.data[:, 0] = np.round(s.data[:, 0], int(digits[0]))
            s.data[:, 1] = np.round(s.data[:, 1], int(digits[1]))

    plotly_obj["data"].extend(
        {
            "name": s.name,
            "x": s.data[:, 0].tolist(),
            "y": s.data[:, 1].tolist(),
            "mode": mode,
            "text": s.labels,
            "type": "scatter",
        }
        for s in series
    )

    if layout_config:
        plotly_obj["layout"] = merge_dicts(plotly_obj["layout"], layout_config)

    return plotly_obj


def create_2d_scatter_series(
    np_row_wise: numpy.ndarray,
    title: str = "Scatter",
    series_name: str = "Series",
    xtitle: str = "x",
    ytitle: str = "y",
    mode: str = "lines",
    labels: Optional[List[str]] = None,
    comment: Optional[str] = None,
    layout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a 2D scatter Plotly graph from a 2 column numpy array
    :param np_row_wise: 2 column numpy data array [(x0,y0), (x1,y1) ...]
    :param title: Chart title
    :param series_name: Series name
    :param xtitle: X-axis title
    :param ytitle: Y-axis title
    :param mode: scatter type mode ('lines' / 'markers' / 'lines+markers')
    :param labels: label (text) per point on the scatter graph
    :param comment: comment underneath the title
    :param layout_config: optional dictionary for layout configuration, passed directly to plotly
    :return: Plotly chart dict.
    """
    plotly_obj = _plotly_scatter_layout_dict(  # noqa: F841
        title=title, xaxis_title=xtitle, yaxis_title=ytitle, comment=comment
    )
    assert np_row_wise.ndim == 2, "Expected a 2D numpy array"
    assert np_row_wise.shape[1] == 2, "Expected two columns X/Y e.g. [(x0,y0), (x1,y1) ...]"

    # this_scatter_data = {
    #     "name": series_name,
    #     "x": np_row_wise[:, 0].tolist(),
    #     "y": np_row_wise[:, 1].tolist(),
    #     "mode": mode,
    #     "text": labels,
    #     "type": "scatter"
    # }
    # plotly_obj["data"].append(this_scatter_data)
    # return plotly_obj
    series = SeriesInfo(name=series_name, data=np_row_wise, labels=labels)

    return create_line_plot(
        title=title,
        series=[series],
        xtitle=xtitle,
        ytitle=ytitle,
        mode=mode,
        comment=comment,
        MAX_SIZE=100000,
        layout_config=layout_config,
    )


def create_3d_scatter_series(
    np_row_wise: numpy.ndarray,
    title: str = "Scatter",
    series_name: str = "Series",
    xtitle: str = "x",
    ytitle: str = "y",
    ztitle: str = "z",
    mode: str = "lines",
    color: Tuple[Tuple[int, int, int, float]] = ((217, 217, 217, 0.14),),
    marker_size: int = 5,
    line_width: float = 0.8,
    labels: List[str] = None,
    fill_axis: int = -1,
    plotly_obj: dict = None,
    layout_config: dict = None,
) -> dict:
    """
    Create a 3D scatter Plotly graph from a 3 column numpy array
    :param np_row_wise: 3 column numpy data array [(x0,y0,z0), (x1,y1,z1) ...]
    :param title: Chart title
    :param series_name: Series name
    :param xtitle: X-axis title
    :param ytitle: Y-axis title
    :param ztitle: Z-axis title
    :param labels: label (text) per point on the scatter graph
    :param fill_axis: fill area under the curve
    :param layout_config: additional layout configuration
    :return: Plotly chart dict.
    """
    if not plotly_obj:
        plotly_obj = plotly_scatter3d_layout_dict(
            title=title, xaxis_title=xtitle, yaxis_title=ytitle, zaxis_title=ztitle
        )
    assert np_row_wise.ndim == 2, "Expected a 2D numpy array"
    assert np_row_wise.shape[1] == 3, "Expected three columns X/Y/Z e.g. [(x0,y0,z0), (x1,y1,z1) ...]"

    c = color[0]
    c = (int(c[0]), int(c[1]), int(c[2]), float(c[3]))
    this_scatter_data = {
        "name": series_name,
        "x": np_row_wise[:, 0].tolist(),
        "y": np_row_wise[:, 1].tolist(),
        "z": np_row_wise[:, 2].tolist(),
        "text": labels,
        "type": "scatter3d",
        "mode": mode,
        "marker": {
            "size": marker_size,
            "line": {
                "color": "rgba(%d, %d, %d, %f.2)" % (c[0], c[1], c[2], c[3]),
                "width": line_width,
            },
            "opacity": 0.8,
        },
    }
    plotly_obj["data"].append(this_scatter_data)

    if layout_config:
        plotly_obj["layout"] = merge_dicts(plotly_obj["layout"], layout_config)

    return plotly_obj


def create_value_matrix(
    np_value_matrix: numpy.ndarray,
    title: str = "Heatmap Matrix",
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    xtitle: str = "X",
    ytitle: str = "Y",
    custom_colors: bool = True,
    series: Optional[str] = None,
    comment: Optional[str] = None,
    yaxis_reversed: bool = False,
    layout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conf_matrix_plot = {
        "data": [
            {
                "x": xlabels,
                "y": ylabels,
                "z": np_value_matrix.tolist(),
                "type": "heatmap",
            }
        ],
        "layout": {
            "showlegend": True,
            "title": title if not comment else (title + "<br><sup>" + comment + "</sup>"),
            "xaxis": {
                "title": xtitle,
            },
            "yaxis": {"title": ytitle},
            "name": series,
        },
    }
    if yaxis_reversed:
        conf_matrix_plot["layout"]["yaxis"]["autorange"] = "reversed"

    if custom_colors and not layout_config:
        scale, bar = _get_z_colorbar_data()
        conf_matrix_plot["data"][0].update({"colorscale": scale})
        conf_matrix_plot["data"][0].update({"colorbar": bar})

    if layout_config:
        conf_matrix_plot["data"][0] = merge_dicts(conf_matrix_plot["data"][0], layout_config)
        conf_matrix_plot["layout"] = merge_dicts(conf_matrix_plot["layout"], layout_config)

    return conf_matrix_plot


def create_3d_surface(
    np_value_matrix: numpy.ndarray,
    title: str = "3D Surface",
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    xtitle: str = "X",
    ytitle: str = "Y",
    ztitle: str = "Z",
    custom_colors: bool = True,
    series: Optional[str] = None,
    camera: Optional[Tuple[float, float, float]] = None,
    comment: Optional[str] = None,
    layout_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    surface_plot = {
        "data": [
            {
                "z": np_value_matrix.tolist(),
                "type": "surface",
                "contours": {
                    "y": {
                        "show": False,
                        "highlightcolor": "#fff4ff",
                        "project": {"y": True},
                    }
                },
                "showscale": False,
            }
        ],
        "layout": {
            "scene": {
                "xaxis": {
                    "title": xtitle,
                    "showgrid": False,
                    "nticks": 10,
                    "ticktext": xlabels,
                    "tickvals": list(range(len(xlabels))) if xlabels else None,
                },
                "yaxis": {
                    "title": ytitle,
                    "showgrid": False,
                    "nticks": 10,
                    "ticktext": ylabels,
                    "tickvals": list(range(len(ylabels))) if ylabels else ylabels,
                },
                "zaxis": {
                    "title": ztitle,
                    "nticks": 5,
                },
            },
            "showlegend": False,
            "title": title if not comment else (title + "<br><sup>" + comment + "</sup>"),
            "name": series,
        },
    }
    if camera:
        surface_plot["layout"]["scene"]["camera"] = {"eye": {"x": camera[0], "y": camera[1], "z": camera[2]}}

    if custom_colors:
        scale, bar = _get_z_colorbar_data()
        surface_plot["data"][0].update({"colorscale": scale})
        surface_plot["data"][0].update({"colorbar": bar})

    if layout_config:
        surface_plot["layout"] = merge_dicts(surface_plot["layout"], layout_config)

    return surface_plot


def create_image_plot(
    image_src: str,
    title: str,
    width: int = 640,
    height: int = 480,
    series: str = None,
    comment: str = None,
    layout_config: dict = None,
) -> dict:
    image_plot = {
        "data": [],
        "layout": {
            "xaxis": {"visible": False, "range": [0, width]},
            "yaxis": {"visible": False, "range": [0, height], "scaleanchor": "x"},
            # "width": width,
            # "height": height,
            "margin": {"l": 0, "r": 0, "t": 64, "b": 0},
            "images": [
                {
                    "sizex": width,
                    "sizey": height,
                    "xref": "x",
                    "yref": "y",
                    "opacity": 1.0,
                    "x": 0,
                    "y": height,
                    # "xanchor": "left",
                    # "yanchor": "bottom",
                    "sizing": "contain",
                    "layer": "below",
                    "source": image_src,
                }
            ],
            "showlegend": False,
            "title": title if not comment else (title + "<br><sup>" + comment + "</sup>"),
            "name": series,
        },
    }

    if layout_config:
        image_plot["layout"] = merge_dicts(image_plot["layout"], layout_config)

    return image_plot


def _get_z_colorbar_data(
    z_data: Optional[Union[np.ndarray, List[List[float]]]] = None,
    values: Optional[List[float]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[List[List[Union[float, str]]], Dict[str, Union[int, str, List[float]]]]:
    if values is None:
        values = [0, 1.0 / 10, 2.0 / 10, 6.0 / 10, 9.0 / 10]
    if colors is None:
        colors = [
            (71, 17, 100),
            (53, 92, 140),
            (37, 139, 141),
            (66, 189, 112),
            (141, 314, 68),
            (221, 226, 24),
        ]
    if z_data is not None:
        data = np.array(z_data)
        max_z = data.max()
        scaler = max_z
        values = [float(v * scaler) for v in values[0:5]]
    values.append(1.0)  # poltly quirk?
    # we do not want to show the first and last value
    tickvalues = [" %.3f " % v for v in values[1:]]
    tickvalues = [float(v) for v in tickvalues]
    # tickvalues.pop()
    colorscale = [[v, "rgb" + str(color)] for v, color in zip(values, colors)]
    colorbar = {"tick0": 0, "tickmode": "array", "tickvals": tickvalues}

    return colorscale, colorbar


def _plotly_hist_dict(
    title: str,
    xtitle: str,
    ytitle: str,
    mode: str = "group",
    data: list = None,
    comment: str = None,
    layout_config: dict = None,
) -> dict:
    """
    Create a basic Plotly chart dictionary
    :param title: Chart title
    :param xtitle: X-Series title
    :param ytitle: Y-Series title
    :param mode: multiple histograms mode. optionals stack / group / relative. Default is 'group'.
    :param data: Data items
    :type data: list
    :param layout_config: dict
    :return: Plotly chart dict.
    """
    assert mode in ("stack", "group", "relative")

    plotly_object = {
        "data": data or [],
        "layout": {
            "title": title if not comment else (title + "<br><sup>" + comment + "</sup>"),
            "xaxis": {"title": xtitle},
            "yaxis": {"title": ytitle},
            "barmode": mode,
            "bargap": 0.08,
            "bargroupgap": 0,
        },
    }
    if layout_config:
        plotly_object["layout"] = merge_dicts(plotly_object["layout"], layout_config)

    return plotly_object


def _np_row_to_plotly_data_item(
    np_row: numpy.ndarray,
    label: str,
    xlabels: Optional[List[Any]] = None,
    data_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a numpy data row into a Plotly chart data item
    :param np_row: numpy 1D data row
    :param label: Item label
    :param xlabels: The labels of the x axis.
    :param dict data_args: Plotly data arguments
    :return: A plotly data item dict.
    """
    bins = list(range(np_row.shape[0])) if xlabels is None else list(xlabels)
    # mylabels = ['"' + label + '"'] * len(bins)
    this_trace_data = {
        "name": label,
        "y": np_row.tolist(),
        "x": bins,
        # "text": mylabels,
        "type": "bar",
    }
    if data_args:
        this_trace_data.update(data_args)
    return this_trace_data


def _plotly_scatter_layout_dict(
    title: str = "Scatter",
    xaxis_title: str = "X",
    yaxis_title: str = "Y",
    series: Optional[str] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "data": [],
        "layout": {
            "title": title if not comment else (title + "<br><sup>" + comment + "</sup>"),
            "xaxis": {
                "title": xaxis_title,
                "showspikes": True,
                "spikethickness": 1,
                "spikesnap": "cursor",
                "spikemode": "toaxis+across",
            },
            "yaxis": {
                "title": yaxis_title,
                "showspikes": True,
                "spikethickness": 1,
                "spikesnap": "cursor",
                "spikemode": "toaxis+across",
            },
            "name": series,
        },
    }


def plotly_scatter3d_layout_dict(
    title: str = "Scatter",
    xaxis_title: str = "X",
    yaxis_title: str = "Y",
    zaxis_title: str = "Z",
    series: str = None,
    show_legend: bool = True,
    comment: str = None,
    layout_config: dict = None,
) -> dict:
    plotly_object = {
        "data": [],
        "layout": {
            "showlegend": show_legend,
            "title": title if not comment else (title + "<br><sup>" + comment + "</sup>"),
            "scene": {
                "xaxis": {"title": xaxis_title},
                "yaxis": {"title": yaxis_title},
                "zaxis": {"title": zaxis_title},
            },
            "name": series,
        },
    }

    if layout_config:
        plotly_object["layout"] = merge_dicts(plotly_object["layout"], layout_config)

    return plotly_object


def create_plotly_table(
    table_plot: Union["pd.DataFrame", List[List[Any]]],
    title: str,
    series: str,
    layout_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a basic Plotly table json style to be sent

    :param table_plot: the output table in pandas.DataFrame structure or list of rows (list) in a table
    :param title: Title (AKA metric)
    :type title: str
    :param series: Series (AKA variant)
    :type series: str
    :param layout_config: additional configuration layout
    :param data_config: additional configuration for the data
    :return: dict with plotly data.
    """
    is_list = isinstance(table_plot, (list, tuple))
    if is_list and (not table_plot or not any(table_plot)):
        # The list if empty
        headers_values = []
        cells_values = []
    elif is_list and table_plot[0] and isinstance(table_plot[0], (list, tuple)):
        headers_values = table_plot[0]
        cells_values = [list(i) for i in zip(*table_plot[1:])]
    else:
        if not pd:
            raise UsageError(
                "pandas is required in order to support reporting tables using CSV or a URL, "
                "please install the pandas python package"
            )
        index_added = not isinstance(table_plot.index, pd.RangeIndex)
        headers_values = list([col] for col in table_plot.columns)
        cells_values = json.loads(table_plot.T.to_json(orient="values", date_format="iso"))
        if index_added:
            if isinstance(table_plot.index, pd.MultiIndex):
                headers_values = [n or "" for n in (table_plot.index.names or [])] + headers_values
                cells_values = list(zip(*(table_plot.index.values.tolist() or []))) + cells_values
            else:
                headers_values.insert(0, table_plot.index.name or "")
                cells_values.insert(0, table_plot.index.values.tolist())

    ret = {
        "data": [
            {
                "type": "table",
                "header": {
                    "values": headers_values,
                    "align": "left",
                    "line": {"width": 0.5, "color": "#d4d6e0"},
                    "fill": {"color": "#fff"},
                    "font": {
                        "family": "Heebo, verdana, arial, sans-serif",
                        "size": 12,
                        "color": "#333",
                    },
                },
                "cells": {
                    "height": 30,
                    "values": cells_values,
                    "align": "left",
                    "line": {"color": "white", "width": 1},
                    "font": {
                        "family": "Heebo, verdana, arial, sans-serif",
                        "size": 14,
                        "color": "#384161",
                    },
                },
            }
        ],
        "layout": {
            "title.text": title,
            "title": series,
        },
    }
    if layout_config:
        ret["layout"] = merge_dicts(ret["layout"], layout_config)

    if data_config and len(ret["data"]) == 1:
        ret["data"][0] = merge_dicts(ret["data"][0], data_config)

    return ret
