import re

import numpy as np
import plotly.graph_objects as go
import plotly.offline
from cxotime import CxoTime

_MATHTEXT_SUP_RE = re.compile(r"\$\^\{(.*?)\}\$")


def mathtext_to_html(text):
    """Convert matplotlib-mathtext superscripts/newlines to plotly's markup."""
    text = _MATHTEXT_SUP_RE.sub(lambda m: f"<sup>{m.group(1)}</sup>", text)
    return text.replace("\n", "<br>")


_MPL_BASE_COLORS = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "k": "black",
    "w": "white",
}

_MPL_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_MPL_TAB_COLORS = {
    "tab:blue": _MPL_CYCLE[0],
    "tab:orange": _MPL_CYCLE[1],
    "tab:green": _MPL_CYCLE[2],
    "tab:red": _MPL_CYCLE[3],
    "tab:purple": _MPL_CYCLE[4],
    "tab:brown": _MPL_CYCLE[5],
    "tab:pink": _MPL_CYCLE[6],
    "tab:gray": _MPL_CYCLE[7],
    "tab:olive": _MPL_CYCLE[8],
    "tab:cyan": _MPL_CYCLE[9],
}

_MPL_DASH = {
    "-": "solid",
    "solid": "solid",
    "--": "dash",
    "dashed": "dash",
    ":": "dot",
    "dotted": "dot",
    "-.": "dashdot",
    "dashdot": "dashdot",
}

_LOC_MAP = {
    "best": {},
    "upper right": {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"},
    "upper left": {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
    "upper center": {"x": 0.5, "y": 0.99, "xanchor": "center", "yanchor": "top"},
    "lower right": {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom"},
    "lower left": {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"},
    "lower center": {"x": 0.5, "y": 0.01, "xanchor": "center", "yanchor": "bottom"},
    "center right": {"x": 0.99, "y": 0.5, "xanchor": "right", "yanchor": "middle"},
    "center left": {"x": 0.01, "y": 0.5, "xanchor": "left", "yanchor": "middle"},
    "center": {"x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle"},
}


def _mpl_color(color):
    if color is None:
        return None
    if color in _MPL_BASE_COLORS:
        return _MPL_BASE_COLORS[color]
    if color in _MPL_TAB_COLORS:
        return _MPL_TAB_COLORS[color]
    if len(color) >= 2 and color[0] == "C" and color[1:].isdigit():
        return _MPL_CYCLE[int(color[1:]) % len(_MPL_CYCLE)]
    return color


def _mpl_linestyle(ls):
    return _MPL_DASH.get(ls, ls)


def _legend_loc(loc):
    return dict(_LOC_MAP.get(loc, {}))


# Chandra "DOY" time format: YYYY:DDD:HH:MM:SS (matches CxoTime's .yday).
DOY_TICKFORMAT = "%Y:%j:%H:%M:%S"

# Zoom-dependent tick label resolution, all in Chandra DOY style: coarse
# ticks (dtick >= 1 day) show only the date, hour-scale ticks add
# hour:minute, and anything finer shows full hour:minute:second.
DOY_TICKFORMATSTOPS = [
    {"dtickrange": [None, 3600000], "value": "%H:%M:%S"},
    {"dtickrange": [3600000, 86400000], "value": "%j %H:%M"},
    {"dtickrange": [86400000, None], "value": "%Y:%j"},
]

# Default figure size for a single-panel time-series plot (taller than
# plotly's default 700x450, which reads as too wide/flat for these plots).
FIGURE_SIZE = {"width": 900, "height": 700}


def _setup_date_xaxis(fig, row=None, col=None):
    fig.update_xaxes(
        title={"text": "Date"},
        tickformatstops=DOY_TICKFORMATSTOPS,
        hoverformat=DOY_TICKFORMAT,
        row=row,
        col=col,
    )


def to_native_endian(arr):
    """Return `arr` as a native-byte-order numpy array.

    FITS/HDF5-backed columns (e.g. GOES data read via astropy) are often
    big-endian; orjson (used by plotly's JSON/HTML serialization) rejects
    non-native-endian arrays outright, so this must run before any array
    reaches a plotly trace.
    """
    arr = np.asarray(arr)
    if arr.dtype.byteorder not in ("=", "|") and arr.dtype.kind in "iuf":
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return arr


class DatePlot:
    """A plotly-backed replacement for acispy's CustomDatePlot/DummyDatePlot."""

    def __init__(
        self,
        dates,
        values,
        *,
        label=None,
        color=None,
        fmt=None,
        ms=6,
        lw=2,
        ls="solid",
        plot=None,
        fig=None,
        row=None,
        col=None,
    ):
        x = to_native_endian(CxoTime(dates).datetime)
        values = to_native_endian(values)

        if plot is not None:
            self.fig = plot.fig
            self._state = plot._state
            self._row = plot._row
            self._col = plot._col
        else:
            if fig is not None:
                self.fig = fig
            else:
                self.fig = go.Figure()
                self.fig.update_layout(**FIGURE_SIZE)
            self._state = {"xlim": None, "color_idx": 0, "yscale": "linear"}
            self._row = row
            self._col = col
            _setup_date_xaxis(self.fig, row=self._row, col=self._col)

        if color is None:
            color = _MPL_CYCLE[self._state["color_idx"] % len(_MPL_CYCLE)]
            self._state["color_idx"] += 1
        else:
            color = _mpl_color(color)

        mode = "markers" if (fmt is not None or lw == 0) else "lines"
        trace_kwargs = {
            "x": x,
            "y": values,
            "mode": mode,
            "name": label,
            "showlegend": label is not None,
        }
        if mode == "markers":
            trace_kwargs["marker"] = {"color": color, "size": ms}
        else:
            trace_kwargs["line"] = {
                "color": color,
                "width": lw,
                "dash": _mpl_linestyle(ls),
            }
        self.fig.add_trace(go.Scatter(**trace_kwargs), row=self._row, col=self._col)

    @classmethod
    def attach(cls, fig, row=None, col=None):
        """Bind to a (sub)plot cell without adding a trace, to seed an overlay chain."""
        self = cls.__new__(cls)
        self.fig = fig
        self._state = {"xlim": None, "color_idx": 0, "yscale": "linear"}
        self._row = row
        self._col = col
        _setup_date_xaxis(self.fig, row=self._row, col=self._col)
        return self

    def set_yscale(self, scale):
        self._state["yscale"] = scale
        self.fig.update_yaxes(
            type="log" if scale == "log" else "linear", row=self._row, col=self._col
        )

    def set_ylabel(self, text, fontsize=18, **kwargs):
        self.fig.update_yaxes(
            title={"text": mathtext_to_html(text), "font": {"size": fontsize}},
            row=self._row,
            col=self._col,
        )

    def set_ylim(self, ymin, ymax):
        if self._state.get("yscale") == "log":
            rng = [np.log10(ymin), np.log10(ymax)]
        else:
            rng = [ymin, ymax]
        self.fig.update_yaxes(range=rng, autorange=False, row=self._row, col=self._col)

    def set_xlim(self, xmin, xmax):
        xmin_dt = CxoTime(xmin).datetime
        xmax_dt = CxoTime(xmax).datetime
        self.fig.update_xaxes(range=[xmin_dt, xmax_dt], row=self._row, col=self._col)
        self._state["xlim"] = (xmin_dt, xmax_dt)

    def get_xlim(self):
        return self._state.get("xlim")

    def set_legend(self, loc="best", fontsize=16, zorder=None, ncols=None, **kwargs):
        legend = {"font": {"size": fontsize}}
        legend.update(_legend_loc(loc))
        if ncols and ncols > 1:
            legend["orientation"] = "h"
        legend.update(kwargs)
        self.fig.update_layout(showlegend=True, legend=legend)

    def add_hline(self, y, lw=2, ls="-", color="green", zorder=None, **kwargs):
        self.fig.add_hline(
            y=y,
            line={"color": _mpl_color(color), "width": lw, "dash": _mpl_linestyle(ls)},
            row=self._row,
            col=self._col,
            **kwargs,
        )

    def add_vline(self, time, lw=2, ls="solid", color="green", zorder=None, **kwargs):
        x = CxoTime(time).datetime
        self.fig.add_vline(
            x=x,
            line={"color": _mpl_color(color), "width": lw, "dash": _mpl_linestyle(ls)},
            row=self._row,
            col=self._col,
            **kwargs,
        )

    def add_vrect(self, x0, x1, color="mediumpurple", label=None, alpha=0.5):
        x0 = CxoTime(x0).datetime
        x1 = CxoTime(x1).datetime
        fill_color = _mpl_color(color)
        self.fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=fill_color,
            opacity=alpha,
            line_width=0,
            row=self._row,
            col=self._col,
        )
        if label is not None:
            self.fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={
                        "size": 10,
                        "symbol": "square",
                        "color": fill_color,
                        "opacity": alpha,
                    },
                    name=label,
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=self._row,
                col=self._col,
            )

    def add_text(
        self,
        time,
        y,
        text,
        *,
        fontsize=18,
        color="black",
        rotation=0,
        yref="data",
        bgcolor=None,
        opacity=1.0,
        horizontalalignment=None,
        zorder=None,
        **kwargs,
    ):
        x = CxoTime(time).datetime
        ann = {
            "x": x,
            "y": y,
            "xref": "x",
            "yref": "y" if yref == "data" else "y domain",
            "text": str(text),
            "showarrow": False,
            "font": {"size": fontsize, "color": _mpl_color(color)},
            "textangle": -rotation,
            "opacity": opacity,
        }
        if bgcolor is not None:
            ann["bgcolor"] = _mpl_color(bgcolor)
        if horizontalalignment is not None:
            ann["xanchor"] = horizontalalignment
        ann.update(kwargs)
        self.fig.add_annotation(row=self._row, col=self._col, **ann)

    def write_html(self, filename, **kwargs):
        self.fig.write_html(filename, **kwargs)


def _as_figure(dp):
    return dp.fig if isinstance(dp, DatePlot) else dp


def write_plotlyjs(filename):
    """Write the plotly.js library to a standalone file.

    Load it once (e.g. `<script src="plotly.min.js"></script>`) on any page
    that embeds fragments from `to_html_fragment`/`write_html_fragment`,
    instead of bundling the ~4.5MB library into every fragment.
    """
    with open(filename, "w") as f:
        f.write(plotly.offline.get_plotlyjs())


def to_html_fragment(dp, **kwargs):
    """Return a figure's embeddable HTML (div + script), without plotly.js.

    For inserting a plot inline into a page that already loads plotly.js
    once itself, e.g. a MyST Markdown document via a raw HTML block.
    """
    return _as_figure(dp).to_html(full_html=False, include_plotlyjs=False, **kwargs)


def write_html_fragment(dp, filename, **kwargs):
    """Write a figure's embeddable HTML fragment (see `to_html_fragment`) to `filename`."""
    with open(filename, "w") as f:
        f.write(to_html_fragment(dp, **kwargs))


def write_image(dp, filename, *, scale=2, **kwargs):
    """Write a figure to a static image file (PNG, PDF, SVG, ...) via kaleido.

    For non-interactive output where the HTML/JS fragment can't be used, e.g.
    a LaTeX/PDF build of a Sphinx doc that also embeds the interactive
    fragment for HTML. `scale` defaults to 2x the figure's nominal pixel
    size (see `FIGURE_SIZE`) for print-quality resolution.
    """
    _as_figure(dp).write_image(filename, scale=scale, **kwargs)


def write_iframe_page(dp, filename, plotlyjs_src="plotly.min.js"):
    """Write a small standalone HTML page suitable as an `<iframe>` src.

    References plotly.js by relative URL (`plotlyjs_src`) instead of
    bundling it, so multiple iframes on one page share a single cached
    download rather than each pulling in the ~4.5MB library.
    """
    frag = to_html_fragment(dp)
    page = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>"
        f"<script src='{plotlyjs_src}'></script></head>"
        f"<body style='margin:0'>{frag}</body></html>"
    )
    with open(filename, "w") as f:
        f.write(page)


def combine_html(sections, filename):
    """Write (title, DatePlot | go.Figure) sections to one self-contained HTML file."""
    js = plotly.offline.get_plotlyjs()
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<script type='text/javascript'>{js}</script>",
        "</head><body>",
    ]
    for title, dp in sections:
        parts.append(f"<h2>{title}</h2>")
        parts.append(to_html_fragment(dp))
    parts.append("</body></html>")
    with open(filename, "w") as f:
        f.write("\n".join(parts))
