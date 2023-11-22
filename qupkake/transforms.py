from typing import List, Union

import torch
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("y_to_indices")
class YToIndices(BaseTransform):
    r"""Normalizes node features to mean :obj:`0` and standard deviation :obj:`1`
    in :obj:`[0, 1]`."""

    def __init__(self):
        super(YToIndices, self).__init__()

    def __call__(self, data):
        y_new = torch.zeros(data.x.shape[0], dtype=torch.float16)
        indices = data.y.reshape(1, -1)[0]
        y_new[indices] = 1.0
        data.y = y_new.reshape(-1, 1)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@functional_transform("to_tensor")
class ToTensor(BaseTransform):
    def __init__(self, attrs: List[str]):
        self.attrs = attrs
        super(ToTensor, self).__init__()

    def __call__(self, data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if isinstance(value, torch.Tensor):
                    store[key] = value.reshape(-1, 1).float()
                else:
                    store[key] = torch.tensor(value, dtype=torch.float).reshape(-1, 1)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@functional_transform("include_energy")
class IncludeEnergy(BaseTransform):
    def __init__(self, attrs: List[str], d_energy: str, index: int):
        self.attrs = attrs
        self.d_energy = d_energy
        self.index = index
        super(IncludeEnergy, self).__init__()

    def __call__(self, data):
        for attr in self.attrs:
            if hasattr(data, attr) and hasattr(data, self.d_energy):
                data[attr][0][self.index] = data[self.d_energy].squeeze()
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"
