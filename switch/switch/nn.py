import itertools
import os
from abc import abstractmethod
from typing import Any, List, Mapping, NoReturn, Optional, Tuple
import pynvml
import numpy as np

import torch
import logging
import torch.distributions as D
import torch.nn.functional as F

from . import model

class SWITCH_nn(torch.nn.Module):

    def __init__(
            self, g2v: model.GraphEncoder, v2g: model.GraphDecoder,
            x2u: Mapping[str, model.DataEncoder],
            u2x: Mapping[str, model.NBDataDecoder],
            idx: Mapping[str, torch.Tensor],
            adj: Mapping[str, torch.Tensor],
            du: model.Discriminator,
            prior: model.Prior,
            logger: logging.StreamHandler,
            normalize_methods: dict
    ) -> None:
        super().__init__()
        if not set(x2u.keys()) == set(u2x.keys()) == set(idx.keys()) != set():
            raise ValueError(
                "`x2u`, `u2x`, `idx` should share the same keys "
                "and non-empty!"
            )
        self.keys = list(idx.keys())  # Keeps a specific order
        self.normalize_methods = normalize_methods

        self.g2v = g2v
        self.v2g = v2g
        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2x = torch.nn.ModuleDict(u2x)
        for k, v in idx.items():  # Since there is no BufferList
            self.register_buffer(f"{k}_idx", v)
        self.du = du
        self.prior = prior
        self.logger = logger
        self.device = self.autodevice()

        self.adj = adj
        self.paired_idx = None
        self.adj_weight = None
        self.filtered_idx1 = None
        self.filtered_idx2 = None

    @property
    def device(self) -> torch.device:

        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)
    
    def autodevice(self) -> torch.device:
        used_device = -1
        gpu_available = torch.cuda.is_available()
        if(torch.cuda.is_available()):
            try:
                pynvml.nvmlInit()
                free_mems = np.array([
                    pynvml.nvmlDeviceGetMemoryInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(i)
                    ).free for i in range(pynvml.nvmlDeviceGetCount())
                ])
                if free_mems.size:
                    best_devices = np.where(free_mems == free_mems.max())[0]
                    used_device = np.random.choice(best_devices, 1)[0]
                    if free_mems[used_device] < 0:
                        used_device = -1
            except pynvml.NVMLError:
                pass
        if used_device == -1:
            self.logger.info(f"GPU available: {str(gpu_available)}, used device: CPU")
            return torch.device("cpu")
        self.logger.info(f"GPU available: {str(gpu_available)}, used device: GPU {used_device}")
        return torch.device(f"cuda:{used_device}")
    
    def forward(self) -> NoReturn:

        raise NotImplementedError