import os, pathlib, bz2, re, time
import yaml
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from . import utils


def plot_dfs(
    dfs,
    markers=None,
    figsize=(6, 3),
    dpi=300,
    legends=None,
    logx=True,
    logy=True,
    metric="DRAM_Accesses",
    shape_name="",
    probs=None,
    motivation=False,
    plot_min=False,
    plot_buf=False,
    plot_all_mappings=False,
    xlim=None,
    ylim=None,
    y_end_value=None,
    plot_gpu_data=False,
    gpu_data=None,
    coefficient=1,
    simba_df=None,
    simba_dfs=None,
    cmap=None,
    legend_fontsize=None,
):
    drawstyle = None  # "steps-post"
    plt.figure(dpi=dpi)
    ax = plt.gca()
    fig = ax.figure

    if plot_buf or plot_min or motivation:
        prob = probs[0]
        size, W_size, I_size, O_size = prob.get_tensor_size()
        op_int = prob.get_op_int()
        compute_size = prob.get_compute_size()

        if metric == "DRAM_Accesses":
            algo_max = compute_size * 3
            ylim = (None, algo_max * 3)
        else:
            op_int_min = compute_size * 2 / (compute_size * 3)
            ylim = (0.1, 5000)

    #    cmap = LinearSegmentedColormap.from_list("c10", plt.rcParams['axes.prop_cycle'].by_key()['color'])

    if cmap is not None:
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)

    norm = mcolors.Normalize(vmin=0, vmax=len(dfs))

    for df_idx, df in enumerate(dfs):
        if cmap is not None:
            color = cmap(norm(df_idx))

        ylabel = metric.replace("_", " ")
        if metric == "DRAM_Accesses":
            ylabel = "Backing Store Accesses (B)"
        else:
            ylabel += " (Op/B)"

        df_plot = df.copy()
        df_plot = df_plot * coefficient
        df_plot.index = df_plot.index * coefficient

        if y_end_value is not None:
            new_row_data = {
                "DRAM_Accesses": df_plot["DRAM_Accesses"].min(),
                "Op_Intensity": df_plot["Op_Intensity"].max(),
            }
            new_row_df = pd.DataFrame([new_row_data], index=[y_end_value])
            df_plot = pd.concat([df_plot, new_row_df])

        if plot_all_mappings:
            ax = df_plot[metric].plot(
                figsize=figsize,
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                xlabel="Buffer Size (B)",
                ylabel=ylabel,
                logx=logx,
                logy=logy,
                alpha=0.7,
                marker=".",
                linestyle="",
                markersize=1,
            )
        else:
            if cmap is None:
                ax = df_plot[metric].plot(
                    drawstyle=drawstyle,
                    figsize=figsize,
                    ax=ax,
                    xlim=xlim,
                    ylim=ylim,
                    xlabel="Buffer Size (B)",
                    ylabel=ylabel,
                    logx=logx,
                    logy=logy,
                    alpha=0.7,
                    markersize=4,
                )
            else:
                ax = df_plot[metric].plot(
                    drawstyle=drawstyle,
                    figsize=figsize,
                    ax=ax,
                    xlim=xlim,
                    ylim=ylim,
                    xlabel="Buffer Size (B)",
                    ylabel=ylabel,
                    logx=logx,
                    logy=logy,
                    alpha=0.7,
                    markersize=4,
                    color=color
                )


    if markers is not None:
        for i, line in enumerate(ax.get_lines()):
            line.set_marker(markers[i])

    if plot_all_mappings:
        ax = df_plot[metric].plot.line(
            figsize=figsize,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            xlabel="Buffer Size (B)",
            ylabel=ylabel,
            logx=logx,
            logy=logy,
            alpha=1,
            color="g",
        )  # color='g'

    if plot_buf:
        if metric == "DRAM_Accesses":
            max_accesses = df.iloc[0][metric]
        else:
            max_accesses = df.iloc[-1][metric]
        for idx, (k, v) in enumerate(
            zip(["total", "W", "I", "O"], [size, W_size, I_size, O_size])
        ):
            line = ax.axvline(x=v, color="black", linestyle=":", lw=1)
            ax.annotate(
                k,
                xy=(v, max_accesses * (0.5**idx)),
                xytext=(v * 1.4, max_accesses * (0.5**idx)),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )

    if plot_min:
        if metric == "DRAM_Accesses":
            min_accesses = df.iloc[-1][metric]
            assert min_accesses == size
            optimal_access = size
            line = ax.axhline(
                y=optimal_access,
                color="r",
                linestyle="--",
                lw=1,
                xmin=-100,
                xmax=size * 10,
            )
            x_point = df.index.min()
            ax.annotate(
                "Algo min",
                xy=(x_point, size),
                xytext=(x_point * 2, optimal_access * 1.1),
                ha="center",
                fontsize=10,
            )

            x_point = df.index.min()
            line = ax.axhline(y=algo_max, color="r", linestyle="--", lw=1)
            ax.annotate(
                "Algo max",
                xy=(x_point, algo_max),
                xytext=(x_point * 2, algo_max * 1.1),
                ha="center",
                fontsize=10,
            )
        else:
            line = ax.axhline(y=op_int, color="r", linestyle="--", lw=1)
            x_point = df.index.min()
            ax.annotate(
                "Algo max",
                xy=(x_point, op_int),
                xytext=(x_point * 2, op_int * 0.6),
                ha="center",
                fontsize=10,
            )

            line = ax.axhline(y=op_int_min, color="r", linestyle="--", lw=1)
            x_point = df.index.min()
            ax.annotate(
                "Algo min",
                xy=(x_point, op_int_min),
                xytext=(x_point * 2, op_int_min * 0.6),
                ha="center",
                fontsize=10,
            )

    if metric == "DRAM_Accesses":
        max_accesses = df.iloc[0][metric]
    else:
        max_accesses = df.iloc[-1][metric]
    if motivation:
        idx = 20  # 25
        x = df.index[idx]
        y = size

        v = df.index.max()
        #         line = ax.axvline(x=v, color='black', linestyle=':', lw=1)
        #         ax.annotate('size', xy=(v, 0), xytext=(v*1.5, 0), fontsize=6, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        #         ax.annotate(k, xy=(v, max_accesses*(1-0.32*idx)), xytext=(v*1.4, max_accesses*(1-0.32*idx)), fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        #         ax.annotate('max effectual', xy=(v, max_accesses*(0.5**1)), xytext=(v*1.4, max_accesses*(0.5**1)), fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        line = ax.axvline(x=1, color="black", linestyle=":", lw=1)
        ax.annotate(
            "No buffer",
            rotation=270,
            annotation_clip=False,
            clip_on=False,
            va="center",
            xy=(1, max_accesses * 0.1),
            xytext=(1, max_accesses * 0.1),
            fontsize=10,
        )  # , arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        #         annotation_clip=False, clip_on=False, /
        line = ax.axvline(x=size, color="black", linestyle=":", lw=1)
        ax.annotate(
            f"Total oprand size",
            rotation=270,
            annotation_clip=False,
            clip_on=False,
            va="center",
            xy=(size, max_accesses * 0.1),
            xytext=(size, max_accesses * 0.1),
            fontsize=10,
        )  # arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    else:
        if plot_min:
            v = size
            #         line = ax.axvline(x=v, color='black', linestyle=':', lw=1)
            #         ax.annotate('size', xy=(v, 0), xytext=(v*1.5, 0), fontsize=6, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            #         ax.annotate(k, xy=(v, max_accesses*(1-0.32*idx)), xytext=(v*1.4, max_accesses*(1-0.32*idx)), fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            #         ax.annotate('max effectual', xy=(v, max_accesses*(0.5**1)), xytext=(v*1.4, max_accesses*(0.5**1)), fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            line = ax.axvline(x=0, color="black", linestyle=":", lw=1)
            ax.annotate(
                "No buffer",
                rotation=270,
                annotation_clip=False,
                clip_on=False,
                va="center",
                xy=(0, max_accesses * 0.1),
                xytext=(0, max_accesses * 0.1),
                fontsize=10,
            )  # , arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            #         annotation_clip=False, clip_on=False, /
            line = ax.axvline(x=v, color="black", linestyle=":", lw=1)
            size_mb = v // 2**20
            ax.annotate(
                f"Total oprand size = {size_mb}MB",
                rotation=270,
                annotation_clip=False,
                clip_on=False,
                va="center",
                xy=(v, max_accesses * 0.03),
                xytext=(v, max_accesses * 0.03),
                fontsize=10,
            )  # arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            v = df.index.max()
            line = ax.axvline(x=v, color="g", lw=1)
            size_mb = v // 2**20
            ax.annotate(
                f"Max effectual size = {size_mb}MB",
                rotation=270,
                annotation_clip=False,
                clip_on=False,
                va="center",
                xy=(v, max_accesses * 0.03),
                xytext=(v, max_accesses * 0.03),
                fontsize=10,
            )  # arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    #         if  metric=='DRAM_Accesses':
    #             plt.annotate('', xy=(x, size), xytext=(x, df.iloc[idx][metric]),
    #                 arrowprops=dict(facecolor='black', arrowstyle='<|-|>', shrinkA=0, shrinkB=0, lw=1.5, mutation_scale=20,),
    #                 )
    #             plt.annotate('Mind the gap', xy=(x, size),xytext=(x/10, df.iloc[idx][metric]/8.5), backgroundcolor="w", fontsize=12)

    #         else:
    #             plt.annotate('', xy=(x, op_int), xytext=(x, df.iloc[idx][metric]),
    #                 arrowprops=dict(facecolor='black', arrowstyle='<|-|>', shrinkA=0, shrinkB=0, lw=1.5, mutation_scale=20,),
    #                 )

    #             plt.annotate('Mind the gap', xy=(x, op_int),xytext=(x/10, op_int/9), backgroundcolor="w", fontsize=12)

    #         plt.arrow(x, y, 0, dy, linewidth=1, color='b', head_length=3*size, length_includes_head=True, shape='full')
    # #     # green arrow
    #         plt.arrow(0.85, 0.5, -0.70, 0, head_width=0.05, head_length=0.03, linewidth=4, color='g', length_includes_head=True)

    if legends:
        first_legend = ax.legend(legends, fontsize=legend_fontsize)
    postfix = "_data" if metric == "DRAM_Accesses" else "_oi"
    # Create some data

    if plot_gpu_data:

        #         x = [2*2**20, 2*2**20, 24*2**20, 24*2**20, 40*2**20, 40*2**20 ]
        #         y = [2.69*2**30, 1.58*2**30, 650*2**20, 561.51*2**20, 533*2**20, 411.06*2**20]
        x = gpu_data["simt"][0]
        y = gpu_data["simt"][1]
        #         x = [i/4 for i in x]
        #         y = [i/4 for i in y]
        ax.scatter(x, y, s=10, color="blue")  # 'o' is the marker shape for circles
        point_archs = ["A2\n2MB", "A30\n24M", "A100\n40MB", "H100\n50MB"]
        xy_offsets = [(0, -30), (-10, 15), (0, 15), (15, 15)]
        # Annotate each point with a smaller font size
        for i, (xi, yi) in enumerate(zip(x, y)):
            point_arch = point_archs[i]
            xy_offset = xy_offsets[i]
            plt.annotate(
                f"{point_arch}",  # Text to display
                xy=(xi, yi),
                textcoords="offset points",  # How to position the text
                xytext=xy_offset,  # Distance from text to points (x,y)
                ha="center",  # Horizontal alignment can be left, right or center
                arrowprops=dict(
                    arrowstyle="->", lw=0.8
                ),  # Customizing arrow properties
                fontsize=7,
            )  # Smaller font size

        # Define custom markers for the second legend
        from matplotlib.lines import Line2D

        custom_markers = [
            Line2D(
                [0],
                [0],
                color="blue",
                marker="o",
                linestyle="None",
                markersize=3,
                label="simt",
            )
        ]
        if "tensor" in gpu_data.keys():
            x = gpu_data["tensor"][0]
            y = gpu_data["tensor"][1]
            ax.scatter(
                x, y, s=10, color="green", marker="x"
            )  # 'o' is the marker shape for circles
            xy_offset = (0, 15)

            plt.annotate(
                "",  # Text to display
                xy=(x[0], y[0]),
                textcoords="offset points",  # How to position the text
                xytext=xy_offset,  # Distance from text to points (x,y)
                ha="center",  # Horizontal alignment can be left, right or center
                arrowprops=dict(
                    arrowstyle="->", lw=0.8
                ),  # Customizing arrow properties
                fontsize=7,
            )
            custom_markers.append(
                Line2D(
                    [0],
                    [0],
                    color="green",
                    marker="x",
                    linestyle="None",
                    markersize=3,
                    label="tensor",
                )
            )
        # Manually add the first legend back without modifying it
        plt.gca().add_artist(first_legend)

        # Create the second legend with custom markers
        plt.legend(handles=custom_markers, loc="upper left", fontsize="small")
        postfix += "_gpu"
    if simba_df is not None:
        scatter = ax.scatter(
            simba_df.index, simba_df[metric], c="orange", cmap="viridis", s=0.05
        )
        #         scatter = ax.scatter(simba_df.index, simba_df[metric], c=simba_df['spatial'], cmap='viridis', s=0.05)
        #         fig.colorbar(scatter, ax=ax, label='spatial factor')

        postfix += "_simba"
    #         line = ax.axvline(x=64*2**10, color='black', linestyle=':', lw=1)
    #         ax.annotate(f'GlobalBuf Size = 64KB',  rotation=270,  annotation_clip=False, clip_on=False, va='center', xy=(size, max_accesses*0.1), xytext=(size,max_accesses*0.1), fontsize=10) #arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    if simba_dfs is not None:
        sizes = [0.125, 1, 8, 64, 512]
        sizes_str = [
            f"{int(size*1024)} B" if size < 1 else f"{size} KB" for size in sizes
        ]
        color_names = ["red", "blue", "green", "purple", "cyan"]

        for simba_idx, simba_df in enumerate(simba_dfs):
            size = sizes[simba_idx] * 1024
            simba_df["buf_cap"] = size
            scatter = ax.scatter(
                simba_df["buf_cap"],
                simba_df[metric],
                color=color_names[simba_idx],
                s=0.05,
            )
        postfix += "_5simba"
        import matplotlib.lines as mlines

        plt.gca().add_artist(first_legend)
        second_legend_handles = [
            mlines.Line2D(
                [],
                [],
                color=color_names[i],
                marker="o",
                markersize=3,
                linestyle="None",
                label=sizes_str[i],
            )
            for i in range(0, 5)
        ]
        plt.legend(handles=second_legend_handles, loc="right", fontsize=8)

    plt.savefig(
        f"figs/orojenesis_op{shape_name}{postfix}.pdf", format="pdf", bbox_inches="tight"
    )

    return ax


def plot_bar_ratios(
    output_dir, probs, legends, fig_name, figsize=(7, 3), sort_ratio=False
):
    plt.set_loglevel("WARNING")
    total_tensor_sizes = []
    computes = []

    stats_files = utils.get_stats_files(output_dir, probs)

    for prob in probs:
        total, W, I, O = prob.get_tensor_size()
        compute = prob.get_compute_size()
        computes.append(compute)
        total_tensor_sizes.append(total)
    dfs = utils.get_dfs(stats_files, get_opt=True)
    full_reuse_tensor_sizes = []
    ratios = []
    for idx, df in enumerate(dfs):
        tensor_size = df.index.max()
        full_reuse_tensor_sizes.append(tensor_size)
        ratio = tensor_size / total_tensor_sizes[idx]
        ratios.append(ratio)
    plt.rcParams["font.family"] = "sans-serif"

    legends.reverse()
    ratios.reverse()

    plt.figure(dpi=300, figsize=figsize)
    ax = plt.gca()

    if sort_ratio:
        array1 = np.array(ratios)
        array2 = np.array(legends)

        # Get the sorted indices of array1
        sorted_indices = np.argsort(array1)
        sorted_array1 = array1[sorted_indices]
        # Use the sorted indices to reorder array2
        sorted_array2 = array2[sorted_indices]
        ratios = sorted_array1.tolist()
        legends = sorted_array2.tolist()

    categories = legends
    percentages = ratios  # Adjust these percentages as needed

    bars = ax.barh(
        categories,
        percentages,
        color="lightgray",
        label="Normalized minimal buffer size to enable full reuse",
        height=0.85,
        edgecolor="black",
        linewidth=0.6,
    )  # hatch='xx'
    # ax.bar_label(percentages, label_type='center', color='black')
    for bar, percentage in zip(bars, percentages):
        if percentage > 0.01:
            t = ax.text(
                bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:,.0%}",
                ha="center",
                va="center",
                fontsize=10,
            )  # , backgroundcolor='white')
        else:
            t = ax.text(
                0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:,.2%}",
                ha="center",
                va="center",
                fontsize=10,
            )  # , backgroundcolor='white')
        t.set_bbox(
            dict(facecolor="white", alpha=1, edgecolor="white", linewidth=0, pad=0.1)
        )

        rest_ratios = [1 - ratio for ratio in ratios]
        rects = ax.barh(
            categories,
            rest_ratios,
            left=percentages,
            color="white",
            height=0.85,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.6,
        )
    plt.xlim(0, 1)
    # for i in range(1, len(categories)):
    #     plt.axhline(y=i - 0.5, color='gray', linestyle='--', linewidth=0.5)

    # Set the bar height for the bar that should reach the edge (100%)
    # bars[-1].set_height(1)
    vals = ax.get_xticks()
    ax.set_xticklabels(["{:,.0%}".format(x) for x in vals])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08))  # , fontsize='small')
    # ax.legend(ncols=2, %bbox_to_anchor=(0, 1),
    #               loc='lower right', fontsize='small')
    plt.savefig(f"figs/{fig_name}.pdf", format="pdf", bbox_inches="tight")


def gen_dims(layer_name, layers_dict):
    dims = layers_dict[layer_name].replace("prob_", "").split("_")
    dims = [int(dim) for dim in dims]
    return dims


def get_optimal_performance(chains, layers_dict, num_heads, batch_size=1):
    optimal_accesses = []
    optimal_accesses_fused = []
    for chain_idx, chain in enumerate(chains):
        optimal_accesses_sub_chain = 0
        optimal_accesses_sub_chain_fused = 0
        for sub_chain_idx, sub_chain in enumerate(chain[1]):
            for item_idx, item in enumerate(sub_chain):
                dims = gen_dims(item, layers_dict)
                prob = item
                input_factor = 1
                output_factor = 1
                weight_factor = 1
                if prob in ["mm_proj_0", "mm_proj_1", "mm_proj_2", "mm_qk", "mm_qkv"]:
                    output_factor = num_heads
                if prob in ["mm_qk", "mm_qkv", "mm_proj_final"]:
                    input_factor = num_heads
                if prob in [
                    "mm_proj_0",
                    "mm_proj_1",
                    "mm_proj_2",
                    "mm_qk",
                    "mm_qkv",
                    "mm_proj_final",
                ]:
                    weight_factor = num_heads
                    if prob in ["mm_qk", "mm_qkv"]:
                        weight_factor *= batch_size
                weight_accesses = dims[1] * dims[2]
                input_accesses = dims[0] * dims[1]
                output_accesses = dims[0] * dims[2]
                total_accesses = 0
                total_accesses_fused = 0

                if item_idx == 0:
                    total_accesses_fused += input_accesses * input_factor
                elif item_idx == len(sub_chain) - 1:
                    total_accesses_fused += output_accesses * output_factor
                total_accesses_fused += weight_accesses * weight_factor
                optimal_accesses_sub_chain_fused += total_accesses_fused

                total_accesses += (
                    weight_accesses * weight_factor
                    + input_accesses * input_factor
                    + output_accesses * output_factor
                )
                optimal_accesses_sub_chain += total_accesses
                # print(f'input {dims[0]} {dims[1]} {input_accesses} {input_factor} {batch_size}')
                # print(f'w,i,o {weight_accesses * weight_factor} {input_accesses * input_factor} {output_accesses * output_factor}')
                # print(prob, dims, total_accesses, total_accesses_fused)

        optimal_accesses.append(optimal_accesses_sub_chain)
        optimal_accesses_fused.append(optimal_accesses_sub_chain_fused)
    return optimal_accesses, optimal_accesses_fused


def plot_accesses_comparison(
    df,
    optimal_access,
    optimal_access_fused,
    figsize=(6, 3),
    df_nochain=None,
    df_nochain_fused=None,
    xbound=(2 * 10**7, 4 * 10**7),
    ybound=(10**8, 10**12),
    df_relax_io=None,
    df_relax_io_kn=None,
    df_slices=None,
    df_slice=None,
    df_spatials=None,
    df_nochain_spatials=None,
    max_effect_size=True,
    plot_gpu_data=False,
    gpu_data=None,
    df_nochain_simba=None,
    df_chain_simba=None,
    legend_fontsize=None,
    y_max_effectual=0.02,
    x_algo_min_unfused=60,
    x_algo_min_fused=55,
    plot_cache=[],
    logx=True,
    logy=True,
):
    drawstype = "steps-post"  # set to "default" for continous line plot
    if df is not None:
        df, df_label = df

        df = df.set_index("max_buf_size").sort_index()
        df["fused_accesses"] = df["fused_accesses"].cummin()
        df["orig_accesses"] = df["orig_accesses"].cummin()
        #     df_access = df[['orig_accesses', 'fused_accesses']]
        df_access = df[["fused_accesses"]]
        #     df_access.columns = ['FOMT w/o fusion', 'FOMT w/ fusion']
        #     df_access.columns = ['FOMT tiled fusion']
        #     df_access.columns = ['FLAT fusion']
        df_access.columns = [df_label]
    fig, ax = plt.subplots(dpi=300)

    # plt.style.use('seaborn-colorblind')
    line1 = ax.axhline(
        y=optimal_access, color="r", linestyle="--", lw=1
    )  # , label='SOL w/o fusion')
    line2 = ax.axhline(
        y=optimal_access_fused, color="b", linestyle="--", lw=1
    )  # , label='SOL w/ fusion')
    #     x_point = df['max_buf_size'].max()
    x_point = xbound[0]  # df.index.max()

    ax.text(
        x_point * x_algo_min_unfused, optimal_access * 1.1,
        "Algo min w/o fusion",
        ha="center",
        fontsize=8,
    )

    ax.text(
        x_point * x_algo_min_fused, optimal_access_fused * 1.1,
        "Algo min w/ fusion",
        ha="center",
        fontsize=8,
    )

    max_accesses = df.iloc[0]["orig_accesses"]
    v = df.index.max()

    if max_effect_size:
        line = ax.axvline(x=v, color="black", lw=1)
        size_mb = v // 2**20
        ax.annotate(
            f"Max effectual size = {size_mb}MB",
            rotation=270,
            annotation_clip=False,
            clip_on=False,
            va="center",
            xy=(v, max_accesses * y_max_effectual),
                                # fused_results['orig_accesses'] = result['weight_accesses'] * weight_access_factor + result['input_accesses'] * input_factor * batch_size + result['output_accesses'] * output_factor * batch_size
            xytext=(v, max_accesses * y_max_effectual),
            fontsize=9,
        )  # arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    h100_pcie = {
        "DRAM": 85899345920,
        "L2+L1": 52428800 + 114 * 228 * 2**10,
        "L2": 52428800,
        "Reg": 29184 * 2**10,
        "L1/SMEM": 114 * 228 * 2**10,
    }
    for idx, (k, v) in enumerate(h100_pcie.items()):
        if k not in plot_cache:
            continue
        line = ax.axvline(x=v, color="black", linestyle=":", lw=1)
        ax.annotate(
            k,
            xy=(v, optimal_access_fused * 20),
            xytext=(v * 0.4, optimal_access_fused * 20),
            fontsize=6,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )

    first_legend = plt.legend(handles=[line1, line2], loc="upper left")
    # Add the first legend manually to the current Axes.
    # plt.gca().add_artist(first_legend)
    if df_spatials is not None:
        for df_spatial_idx, df_spatial in enumerate(df_spatials):
            df_spatial = df_spatial.set_index("max_buf_size").sort_index()
            df_spatial["fused_accesses"] = df_spatial["fused_accesses"].cummin()
            df_spatial["orig_accesses"] = df_spatial["orig_accesses"].cummin()
            df_spatial_access = df_spatial[["orig_accesses", "fused_accesses"]]
            spatial_str = 2 ** (2**df_spatial_idx)
            df_spatial_access.columns = [
                f"FOMT w/o fusion s{spatial_str}",
                f"FOMT w/ fusion s{spatial_str}",
            ]
            df_spatial_access.plot(ax=ax, drawstyle=drawstype, alpha=0.8)
    if df_nochain is not None:
        df_nochain, df_nochain_label = df_nochain
        df_nochain = df_nochain.set_index("max_buf_size").sort_index()
        df_nochain["orig_accesses"] = df_nochain["orig_accesses"].cummin()
        df_nochain_access = df_nochain[["orig_accesses"]]
        df_nochain_access.columns = [df_nochain_label]
        df_nochain_access.plot(
            ax=ax, drawstyle=drawstype, alpha=0.8, color="blueviolet"
        )
    if df_nochain_spatials is not None:
        for df_nochain_spatial_idx, df_nochain_spatial in enumerate(
            df_nochain_spatials
        ):
            df_nochain_spatial = df_nochain_spatial.set_index(
                "max_buf_size"
            ).sort_index()
            df_nochain_spatial["orig_accesses"] = df_nochain_spatial[
                "orig_accesses"
            ].cummin()
            df_nochain_spatial_access = df_nochain_spatial[["orig_accesses"]]
            spatial_str = 2 ** (df_nochain_spatial_idx + 1)
            df_nochain_spatial_access.columns = [f"OPT w/o fusion s{spatial_str}"]
            df_nochain_spatial_access.plot(ax=ax, drawstyle=drawstype, alpha=0.8)
    if df_nochain_fused is not None:
        df_nochain_fused, df_nochain_fused_label = df_nochain_fused
        df_nochain_fused = df_nochain_fused.set_index("max_buf_size").sort_index()
        df_nochain_fused["orig_accesses"] = df_nochain_fused["orig_accesses"].cummin()
        df_nochain_access = df_nochain_fused[["orig_accesses"]]
        df_nochain_access.columns = [
            df_nochain_fused_label
        ]  # ['OPT untiled fusion + nored']
        df_nochain_access.plot(ax=ax, drawstyle=drawstype, alpha=0.8)
    if df is not None:
        df_access.plot(
            ax=ax,
            drawstyle=drawstype,
            figsize=figsize,
            logx=logx,
            logy=logy,
            xlabel="Buffer Size(B)",
            ylabel="Backing Store Accesses(2B)",
            alpha=1,
            color="yellowgreen",
        )

    if df_slice is not None:
        df_slice, df_slice_label = df_slice

        df_slice = df_slice.set_index("max_buf_size").sort_index()
        df_slice["fused_accesses"] = df_slice["fused_accesses"].cummin()
        df_slice_accesses = df_slice[["fused_accesses"]]
        df_slice_accesses.columns = [df_slice_label]
        df_slice_accesses.plot(
            ax=ax, drawstyle=drawstype, alpha=0.7, color="orange", linestyle="-", linewidth=3
        )  # gold

    if df_slices is not None:
        cmap = plt.get_cmap("summer")
        norm = mcolors.Normalize(vmin=0, vmax=len(df_slices))

        for idx, df_slice_sub in enumerate(df_slices):
            df_slice_sub = df_slice_sub.set_index("max_buf_size").sort_index()
            df_slice_sub["fused_accesses"] = df_slice_sub["fused_accesses"].cummin()
            df_slice_accesses = df_slice_sub[["fused_accesses"]]
            slice_name = df_slice_sub.iloc[0]["slice"]
            df_slice_accesses.columns = [f"Slice{idx}: {slice_name}"]
            df_slice_accesses.plot(
                ax=ax, drawstyle=drawstype, color=cmap(norm(idx))
            )  # , cmap='tab10', alpha=0.2+idx*0.02)
        # Modify the colormap properties of each LineCollection object
    #    cmap = plt.cm.get_cmap("plasma")  # Change to a different colormap
    #         for line in lines:
    #             line.set_cmap(cmap)
    if df_relax_io is not None:
        df_relax_io, df_relax_io_label = df_relax_io
        df_relax_io = df_relax_io.set_index("max_buf_size").sort_index()
        df_access = df_relax_io[["fused_accesses"]]
        df_access.columns = [
            df_relax_io_label
        ]  # ['FOMT w/fusion partial K0 and partial Nn ']
        df_access.plot(ax=ax, drawstyle=drawstype, alpha=0.8)

    if df_relax_io_kn is not None:
        df_relax_io_kn, df_relax_io_kn_label = df_relax_io_kn
        df_relax_io_kn = df_relax_io_kn.set_index("max_buf_size").sort_index()
        df_access = df_relax_io_kn[["fused_accesses"]]
        df_access.columns = [
            df_relax_io_kn_label
        ]  # ['FOMT w/ fusion partial K0, N0,  K1, Nn']
        df_access.plot(ax=ax, drawstyle=drawstype, alpha=0.8)


    if df is not None:
        ax.set_xbound(xbound)
        ax.set_ybound(ybound)

    ax.set_xlabel("Buffer Size(B)")
    if plot_gpu_data:
        x = gpu_data["shmem"][0]
        y = gpu_data["shmem"][1]
        ax.scatter(x, y, s=10, color="blue")  # 'o' is the marker shape for circles
        point_archs = ["A100\n17.6MB", "A30\n9M"]
        xy_offsets = [
            (10, 10),
            (0, 20),
        ]
        # Annotate each point with a smaller font size
        for i, (xi, yi) in enumerate(zip(x, y)):
            point_arch = point_archs[i]
            xy_offset = xy_offsets[i]
            plt.annotate(
                f"{point_arch}",  # Text to display
                xy=(xi, yi),
                textcoords="offset points",  # How to position the text
                xytext=xy_offset,  # Distance from text to points (x,y)
                ha="center",  # Horizontal alignment can be left, right or center
                arrowprops=dict(
                    arrowstyle="->", lw=0.8
                ),  # Customizing arrow properties
                fontsize=5,
            )  # Smaller font size

        # Define custom markers for the second legend
        from matplotlib.lines import Line2D

        custom_markers = [
            Line2D(
                [0],
                [0],
                color="blue",
                marker="o",
                linestyle="None",
                markersize=3,
                label="shmem",
            )
        ]
        if False and "rf" in gpu_data.keys():
            x = gpu_data["rf"][0]
            y = gpu_data["rf"][1]
            ax.scatter(
                x, y, s=10, color="green", marker="x"
            )  # 'o' is the marker shape for circles
            custom_markers.append(
                Line2D(
                    [0],
                    [0],
                    color="green",
                    marker="x",
                    linestyle="None",
                    markersize=3,
                    label="rf",
                )
            )
    if df_nochain_simba is not None:
        ax.set_xbound(xbound)
        ax.set_ybound(ybound)
        color_names = ["blue", "orange"]
        scatter = ax.scatter(
            df_nochain_simba["max_buf_size"],
            df_nochain_simba["orig_accesses"],
            c=color_names[0],
            s=0.3,
        )  # s=0.05)
    if df_chain_simba is not None:
        ax.set_xbound(xbound)
        ax.set_ybound(ybound)
        scatter = ax.scatter(
            df_chain_simba["max_buf_size"],
            df_chain_simba["fused_accesses"],
            c=color_names[1],
            marker="o",
            s=1,
        )  # s=0.05)

    return ax


def plot_weight_util_heatmap(df):
    from ipywidgets.widgets import interact_manual
    from matplotlib.colors import LogNorm

    interlayer_weight_util_str = df["interlayer_weight_util"]
    interlayer_weight_util_2d = []
    for interlayer_weight in interlayer_weight_util_str:
        interlayer_weight = interlayer_weight.replace("[", "").replace("]", "")
        weight_buf_list = [int(value) for value in interlayer_weight.split(",")]
        interlayer_weight_util_2d.append(weight_buf_list)
    labels = [
        f"{v[0]:.2e}, M1={v[1]}, M0={32768//v[1]}, accesses={v[2]:.2e}"
        for v in zip(df.index, df["M1"], df["orig_accesses"])
    ]
    if len(interlayer_weight_util_2d) > 0:
        fig, ax = plt.subplots(figsize=(4, 8), dpi=300)
        s = sns.heatmap(
            interlayer_weight_util_2d,
            cmap="viridis",
            linewidths=0.1,
            norm=LogNorm(),
            yticklabels=labels,
        )
        s.set_ylabel("Buf Capacity (B)")
        s.set_xlabel("Layer Index")
        plt.show()


def plot_buf_area_tradeoff(
    mem_bound_perf,
    compute_bound_perf,
    perf,
    buf_ratio,
    intersection,
    area_per_B,
    total_area,
    plot_offset=0,
    figname="design_llm.pdf",
    plot_highlight=True,
    plot_compute_bound=True,
    plot_mem_bound=True,
    plot_best_ratio=True,
):
    best_buf_area = intersection * area_per_B
    best_mac_area = total_area - best_buf_area
    best_ratio = best_buf_area / total_area

    plt.figure(dpi=300, figsize=(2.5, 2.5))
    ax = plt.gca()
    unit = 10**12
    best_buf_size = intersection // 10**6
    if plot_highlight:
        plt.plot(
            buf_ratio,
            perf / unit,
            label=r"Actual Bound",
            color="#2ca02c",
            alpha=0.5,
            linewidth=6,
        )
    if plot_compute_bound:
        plt.plot(
            buf_ratio, compute_bound_perf / unit, label=r"Compute-limited", color="#1f77b4"
        )

    if plot_mem_bound:
        plt.plot(buf_ratio, mem_bound_perf / unit, label=r"Memory-limited", color="#ff7f0e")
    plt.xlabel("Buffer Area Ratio")
    plt.ylabel("Performance (TOps)")
    if plot_best_ratio:
        plt.axvline(x=best_ratio, color="k", linestyle="--", linewidth=1)
        plt.text(
            best_ratio + plot_offset,
            0,
            f"ratio={best_ratio:.2f}\nsize={best_buf_size:.0f}MB",
            verticalalignment="bottom",
        )
    # ax.legend(fontsize=8)
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig(f"figs/{figname}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    print(
        f"Optimal design point buf_area={best_buf_area:.0f}um^2, mac_area={best_mac_area:.0f}um^2, buffer_size={intersection//10**6}MB"
    )
