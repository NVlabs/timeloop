# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import datetime
from typing import Dict, Optional
import pandas as pd
import plotly.express as px
from util import *
import paretos
from typing import List, Tuple
from mappings import FusedMapping
from operations import OperationList

# import dash
# from dash import html

# app = dash.Dash(__name__)


def plot(
    solutions: Union[List[FusedMapping], Dict[str, List[FusedMapping]]],
    title: Optional[str] = "",
    extra_points: List[Tuple[float, float]] = (),
    max_cap_vline=None
):
    if isinstance(solutions, list):
        labels = [""]
        solutions = [solutions]

    else:
        labels = list(solutions.keys())
        solutions = list(solutions.values())

    to_plot = []
    for label, sol in zip(labels, solutions):
        for i, s in enumerate(sol):
            ops = sorted(p.name for p in s.op)
            ljust = max([len(p) for p in ops]) + 1
            df = s.mapping_result.df.copy()

            def make_info_str(row):
                result = []
                for op in ops:
                    util = row[f"{op} {paretos.UTIL_COL}"]
                    acc = row[f"{op} {paretos.ACCESSES_COL}"]
                    tiling = row[f"{op} {paretos.TILING_COL}"]
                    mapping = row[f"{op} {paretos.MAPPING_COL}"]
                    result.append(
                        (
                            util,
                            acc,
                            f"{op.ljust(ljust)} {util:.2e}U {acc: .2e}A {tiling} {mapping}",
                        )
                    )

                result = [r[-1] for r in sorted(result, key=lambda r: (-r[0], -r[1]))]

                return "<br>".join([f"Solution {i}:"] + result)

            df[paretos.TILING_COL] = df.apply(make_info_str, axis=1)
            df["__plotby__"] = f"{label} {i}"
            to_plot.append(df)

    df = pd.concat([df.reset_index(drop=True) for df in to_plot])

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Plot the Pareto
    fig = px.line(
        df,
        title=title or date,
        x=paretos.UTIL_COL,
        y=paretos.ACCESSES_COL,
        color="__plotby__",
        hover_name=paretos.TILING_COL,
        # size=[len(df) for df in [pareto]],
        log_x=True,
        log_y=True,
        markers=True,
    )
    for p in extra_points:
        fig.add_vline(x=max(1, p[0]), line_dash="dot", line_color="red")
        fig.add_hline(y=max(1, p[1]), line_dash="dot", line_color="red")

    if max_cap_vline is not None:
        fig.add_vline(x=max_cap_vline, line_dash="dot", line_color="black")
    fig.layout.xaxis.title = "L2 Footprint (bits)"
    fig.layout.yaxis.title = "DRAM Accesses (bits)"
    fig.update_layout(
        font_family="Courier New, monospace",
        font_color="black",
        hoverlabel=dict(font=dict(family="Courier New, monospace")),
    )

    # pareto.to_excel("timeloop-mapper.oaves.xlsx")
    # fig.show(renderer="browser")
    plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    figpath = os.path.join(plots_dir, f"{title} {date}.html")
    fig.write_html(figpath, auto_open=True)
    return open(figpath).read()


def plot_fancy(
    operations: OperationList,
    solutions: Union[List[FusedMapping], Dict[str, List[FusedMapping]]],
    title: Optional[str] = "",
    extra_points: List[Tuple[float, float]] = (),
    max_cap_vline=None
):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots")
    svg_dir = os.path.join(plots_dir, f"{title} {date}")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    figpath = os.path.join(plots_dir, f"{title} {date}.html")

    if isinstance(solutions, list):
        labels = [""]
        solutions = [solutions]

    else:
        labels = list(solutions.keys())
        solutions = list(solutions.values())

    to_plot = []
    j = 0
    for label, sol in tqdm.tqdm(
        list(zip(labels, solutions)), desc="Generating mapping images"
    ):
        for i, s in enumerate(sol):
            df = s.df.sort_values(by=paretos.UTIL_COL).reset_index(drop=True).copy()

            def make_info_str(row, n):
                d = os.path.abspath(os.path.join(svg_dir, f"{n}.svg"))
                with open(d, "w") as f:
                    f.write(
                        operations.to_pydot_acc_util(f"{label} {n}", row)
                        .create_svg()
                        .decode()
                    )
                return d

            df[paretos.TILING_COL] = parallel_proc(
                [delayed(make_info_str)(row, j + n) for n, row in df.iterrows()],
                pbar=f"Generating images for {label}",
                leave=False,
            )
            df["__plotby__"] = f"{label} {i}"
            to_plot.append(df)
            j += len(df)

    df = pd.concat([df.reset_index(drop=True) for df in to_plot])

    # Plot the Pareto
    fig = px.line(
        df,
        title=title or date,
        x=paretos.UTIL_COL,
        y=paretos.ACCESSES_COL,
        color="__plotby__",
        hover_name=paretos.TILING_COL,
        # size=[len(df) for df in [pareto]],
        log_x=True,
        log_y=True,
        markers=True,
    )
    for p in extra_points:
        fig.add_vline(x=max(1, p[0]), line_dash="dot", line_color="red")
        fig.add_hline(y=max(1, p[1]), line_dash="dot", line_color="red")

    if max_cap_vline is not None:
        fig.add_vline(x=max_cap_vline, line_dash="dot", line_color="black")
    fig.layout.xaxis.title = "L2 Footprint (bits)"
    fig.layout.yaxis.title = "DRAM Accesses (bits)"
    fig.update_layout(
        font_family="Courier New, monospace",
        font_color="black",
        hoverlabel=dict(font=dict(family="Courier New, monospace")),
    )

    fig.write_html(figpath, auto_open=True)

    from IPython.display import SVG, display
    import ipywidgets as widgets

    def show_svg(f):
        display(SVG(os.path.join(svg_dir, f)))

    files = list(df[paretos.TILING_COL])

    return open(figpath).read(), widgets.interactive(
        show_svg,
        f=widgets.SelectionSlider(
            options=[(os.path.basename(f), f) for f in files],
            layout=widgets.Layout(width="95%"),
            description="",
        ),
    )


def plot_bars(
    operations: Union[OperationList, Dict[str, OperationList]],
    solutions: Union[List[FusedMapping], Dict[str, List[FusedMapping]]],
    title: Optional[str] = "",
    max_cap=65536 * 1024 * 8,
):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "plots")
    svg_dir = os.path.join(plots_dir, f"{title} {date}")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    figpath = os.path.join(plots_dir, f"{title} {date}.html")

    if isinstance(solutions, list):
        labels = [""]
        solutions = [solutions]

    else:
        labels = list(solutions.keys())
        solutions = list(solutions.values())

    ops_dict = operations if isinstance(operations, dict) else {"": operations}
    operations = next(iter(ops_dict.values()))

    to_plot = []
    j = 0
    for label, sol in zip(labels, solutions):
        for i, s in enumerate(sol):
            row = s.df.sort_values(by=paretos.UTIL_COL).reset_index(drop=True).copy()
            row = row[row[paretos.UTIL_COL] < max_cap]
            row = row.loc[row[paretos.ACCESSES_COL].idxmin()].copy()

            def make_info_str(r):
                nonlocal j
                d = os.path.abspath(os.path.join(svg_dir, f"{j}.svg"))
                with open(d, "w") as f:
                    f.write(
                        operations.to_pydot_acc_util(f"{label} {j}", r)
                        .create_svg()
                        .decode()
                    )
                j += 1
                return d

            row[paretos.TILING_COL] = make_info_str(row)
            row["__plotby__"] = f"{label} {i}" if len(sol) > 1 else label
            to_plot.append(row)

    baseline = [
        pd.Series(
            {
                paretos.TILING_COL: f"Baseline",
                "__plotby__": f"{k} Baseline" if k else "Baseline",
                **{
                    f"{op.name} {paretos.ACCESSES_COL}": op.baseline_accesses
                    for op in ops
                },
            }
        )
        for k, ops in ops_dict.items()
    ]
    to_plot = baseline + to_plot

    df = pd.DataFrame(to_plot)
    df = df.sort_values("__plotby__").reset_index(drop=True)
    df = df.rename(columns=lambda x: x.replace(f" {paretos.ACCESSES_COL}", ""))

    for op in operations:
        assert op.name in df.columns
    df = df.loc[:, (df != 0).any(axis=0)]

    # Plot the Pareto
    fig = px.bar(
        df,
        title=title or date,
        x="__plotby__",
        y=[op.name for op in operations if op.name in df.columns],
        # color="__plotby__",
        hover_name=paretos.TILING_COL,
        # size=[len(df) for df in [pareto]],
        # log_y=True,
    )

    fig.layout.xaxis.title = "Regime"
    fig.layout.yaxis.title = "DRAM Accesses (bits)"
    fig.update_layout(
        font_family="Courier New, monospace",
        font_color="black",
        hoverlabel=dict(font=dict(family="Courier New, monospace")),
    )

    fig.write_html(figpath, auto_open=True)

    from IPython.display import SVG, display
    import ipywidgets as widgets

    def show_svg(f):
        display(SVG(os.path.join(svg_dir, f)))

    files = list(df[paretos.TILING_COL])
    for f in files:
        print(f)

    return open(figpath).read(), widgets.interactive(
        show_svg,
        f=widgets.SelectionSlider(
            options=[(os.path.basename(f), f) for f in files if f != "Baseline"],
            layout=widgets.Layout(width="95%"),
            description="",
        ),
    )
