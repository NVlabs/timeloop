import argparse
import pytimeloop.timeloopfe.v4 as tl


def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--clear-outputs",
        default=False,
        action="store_true",
        help="Clear all generated outputs",
    )
    argparser.add_argument(
        "--architecture",
        type=str,
        default="eyeriss_like",
        help="Architecture to run in the example_designs directory. "
        "If 'all' is given, all architectures will be run.",
    )
    argparser.add_argument(
        "--generate-ref-outputs",
        default=False,
        action="store_true",
        help="Generate reference outputs instead of outputs",
    )
    argparser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Problem to run in the layer_shapes directory. If a directory is "
        "specified, all problems in the directory will be run. If not specified, "
        "the default problem will be run.",
    )
    argparser.add_argument(
        "--n_jobs", type=int, default=None, help="Number of jobs to run in parallel"
    )
    argparser.add_argument(
        "--remove-sparse-opts",
        default=False,
        action="store_true",
        help="Remove sparse optimizations",
    )
    return argparser.parse_args()


def remove_sparse_optimizations(spec: tl.Specification):
    """This function is used by some Sparseloop tutorials to test with/without
    sparse optimizations"""
    for s in spec.get_nodes_of_type(
        (
            tl.sparse_optimizations.ActionOptimizationList,
            tl.sparse_optimizations.RepresentationFormat,
            tl.sparse_optimizations.ComputeOptimization,
        )
    ):
        s.clear()
    return spec
