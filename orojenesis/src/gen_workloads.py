import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pathlib
import yaml
import copy
from dataclasses import dataclass
from abc import ABC, abstractmethod
import subprocess
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s - (%(filename)s:%(lineno)d)"
)

def store_yaml(yaml_path, data):
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)


def parse_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.full_load(f)
        return data


class Op(ABC):
    "Template for defining Einsum."

    @abstractmethod
    def to_str(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def get_tensor_size(self):
        pass

    @abstractmethod
    def get_compute_size(self):
        pass

    def to_yaml(self, yaml_path):
        prob = self.to_dict()
        logger.info(f"GenProblemYAML> {yaml_path}")
        store_yaml(yaml_path, prob)

    def get_op_int(self):
        return self.get_compute_size() * 2 / self.get_tensor_size()[0]

@dataclass
class GBMMConv(Op):
    "Timeloop Grouped Batch Matrix Multiplication problem definition. Note that the compute reported for grouped Einsum is incorrect."
    P: int = 1
    C: int = 1
    N: int = 1
    H: int = 1
    G: int = 1
    K: int = 1
    Q: int = 1
    R: int = 1
    S: int = 1

    def to_str(self):
        return f"gbmmconv_{self.P}_{self.C}_{self.K}_{self.H}_{self.G}"

    def to_dict(self):
        template = "../configs/gbmmconv_template.yaml"
        d = parse_yaml(template)
        d["problem"]["instance"] = self.__dict__
        return d

    def get_tensor_size(self):
        W = self.C * self.K * self.H // self.G
        I = self.P * self.C * self.H
        O = self.P * self.K * self.H
        return W + I + O, W, I, O

    def get_compute_size(self):
        return self.P * self.C * self.K * self.H


@dataclass
class Conv(Op):
    "Timeloop Conv problem definition"
    R: int = 1
    S: int = 1
    P: int = 1
    Q: int = 1
    C: int = 1
    K: int = 1
    N: int = 1
    Wstride: int = 1
    Hstride: int = 1
    Wdilation: int = 1
    Hdilation: int = 1

    def to_str(self):
        return f"{self.R}_{self.S}_{self.P}_{self.Q}_{self.C}_{self.K}_{self.N}_{self.Wstride}_{self.Hstride}_{self.Wdilation}_{self.Hdilation}"

    def to_dict(self):
        d = {"problem": self.__dict__}
        d["problem"]["shape"] = "cnn-layer"
        return d

    def get_tensor_size(self):
        W = self.R * self.S * self.C * self.K
        P_in = (self.P - 1) * self.Wstride + self.R
        Q_in = (self.Q - 1) * self.Hstride + self.S
        I = P_in * Q_in * self.C * self.N
        O = self.P * self.Q * self.K * self.N
        return W + I + O, W, I, O

    def get_compute_size(self):
        return self.R * self.S * self.P * self.Q * self.C * self.K * self.N


if __name__ == "__main__":

    workloads = [
        {
            "name": "llama2_70b",
            "tp": 1,
            "bs": 1024,
            "in_len": 128,
            "out_len": 1,
            "total_heads": 64,
            "groups": 8,
            "d": 8192,
            "d_hidden": 14336,
        }
    ]

    for workload_id, workload in enumerate(workloads):
        # https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/README.md
        # Note that the KV hidden dimension is derived by the number of KV heads times hidden dimension of each head.
        # LLaMA v2 70B has hidden dimension of 8192, and uses grouped-query attention where 8 key heads and
        # 8 value heads are associated with 64 query heads. Each head has hidden dimension of 8192/64 = 128.
        # So the hidden dimension for KV in total is 128 * 8 * 2 = 2048.
        # For LLaMA v2 70B, there is a restriction on tensor parallelism that the number of KV heads must be divisible
        # by the number of GPUs.
        # For example, since the 70B model has 8 KV heads, you can run it with 2, 4 or 8 GPUs (1 GPU as well for FP8).
        total_d = workload["d"]
        total_heads = workload["total_heads"]
        groups = workload["groups"]
        tp = workload["tp"]
        bs = workload["bs"]
        in_len = workload["in_len"]
        out_len = workload["out_len"]
        context_len = workload["in_len"] + workload["out_len"] // 2
        d_hidden = workload["d_hidden"]

        workload_config_str = workload["name"] + f"_tp{tp}_bs{bs}_{in_len}to{out_len}"
        val_output_dir = pathlib.Path(
            f"../outputs/{workload_config_str}/"
        )
        # val_output_dir = pathlib.Path(f'./outputs/test')
        val_output_dir.mkdir(exist_ok=True, parents=True)

        # prefilling
        d = total_d // tp
        heads = total_heads // tp

        probs = {
            "prefilling": [
                         Conv(P=bs*in_len, C=d, K=d), # projection for q
                         Conv(P=bs*in_len, C=d, K=d * groups * 2 // heads), # project for kv
                         GBMMConv(H=heads*bs, P=in_len, C=d//heads, K=in_len, G=groups / heads),
                         GBMMConv(H=heads*bs, P=in_len, C=in_len, K=d//heads, G=groups / heads),
                         Conv(P=bs * in_len, C=d, K=d),
                         Conv(P=bs*in_len, C=d, K=d_hidden),
                         Conv(P=bs*in_len, C=d_hidden, K=d)
            ],
            "decoding": [
                         Conv(P=bs*1, C=d, K=d),
                         Conv(P=bs*1, C=d, K=d * groups * 2 // heads),
                         GBMMConv(H=heads*bs, P=1, C=d//heads, K=context_len, G=groups / heads),
                         GBMMConv(H=heads*bs, P=1, C=context_len, K=d//heads, G=groups/ heads),
                         Conv(P=bs*1, C=d, K=d),
                         Conv(P=bs*1, C=d, K=d_hidden),
                         Conv(P=bs*1, C=d_hidden, K=d)
            ],
        }


        phases = ["prefilling", 'decoding']
        for phase in phases:
            for prob in probs[phase]:
                print(prob.to_str())
                output_subdir = val_output_dir / f'{phase}' / prob.to_str()
                output_subdir.mkdir(exist_ok=True, parents=True)
                workload_yaml = output_subdir / "problem.yaml"
                prob.to_yaml(workload_yaml)

