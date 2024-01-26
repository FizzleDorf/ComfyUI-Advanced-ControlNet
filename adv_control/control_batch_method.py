import math
import numpy as np
from typing import Iterable, Union
import torch
import torch as th
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat

class BatchSpreadMethod(SparseMethod):
    UNIFORM = "uniform"
    STARTING = "starting"
    ENDING = "ending"
    CENTER = "center"

    LIST = [UNIFORM, STARTING, ENDING, CENTER]

    def __init__(self, spread=UNIFORM):
        super().__init__(self.SPREAD)
        self.spread = spread

    def get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        # if hint_length >= full_length, limit hints to full_length
        if hint_length >= full_length:
            return list(range(full_length))
        # handle special case of 1 hint image
        if hint_length == 1:
            if self.spread in [self.UNIFORM, self.STARTING]:
                return [0]
            elif self.spread == self.ENDING:
                return [full_length-1]
            elif self.spread == self.CENTER:
                # return second (of three) values as the center
                return [np.linspace(0, full_length-1, 3, endpoint=True, dtype=int)[1]]
            else:
                raise ValueError(f"Unrecognized spread: {self.spread}")
        # otherwise, handle other cases
        if self.spread == self.UNIFORM:
            return list(np.linspace(0, full_length-1, hint_length, endpoint=True, dtype=int))
        elif self.spread == self.STARTING:
            # make split 1 larger, remove last element
            return list(np.linspace(0, full_length-1, hint_length+1, endpoint=True, dtype=int))[:-1]
        elif self.spread == self.ENDING:
            # make split 1 larger, remove first element
            return list(np.linspace(0, full_length-1, hint_length+1, endpoint=True, dtype=int))[1:]
        elif self.spread == self.CENTER:
            # if hint length is not 3 greater than full length, do STARTING behavior
            if full_length-hint_length < 3:
                return list(np.linspace(0, full_length-1, hint_length+1, endpoint=True, dtype=int))[:-1]
            # otherwise, get linspace of 2 greater than needed, then cut off first and last
            return list(np.linspace(0, full_length-1, hint_length+2, endpoint=True, dtype=int))[1:-1]
        return ValueError(f"Unrecognized spread: {self.spread}")


class BatchIndexMethod(IndexMethod):
    def __init__(self, idxs: list[int]):
        super().__init__(self.INDEX)
        self.idxs = idxs

    def get_indexes(self, hint_length: int, full_length: int) -> list[int]:
        orig_hint_length = hint_length
        if hint_length > full_length:
            hint_length = full_length
        # if idxs is less than hint_length, throw error
        if len(self.idxs) < hint_length:
            err_msg = f"There are not enough indexes ({len(self.idxs)}) provided to fit the usable {hint_length} input images."
            if orig_hint_length != hint_length:
                err_msg = f"{err_msg} (original input images: {orig_hint_length})"
            raise ValueError(err_msg)
        # cap idxs to hint_length
        idxs = self.idxs[:hint_length]
        new_idxs = []
        real_idxs = set()
        for idx in idxs:
            if idx < 0:
                real_idx = full_length+idx
                if real_idx in real_idxs:
                    raise ValueError(f"Index '{idx}' maps to '{real_idx}' and is duplicate - indexes in Sparse Index Method must be unique.")
            else:
                real_idx = idx
                if real_idx in real_idxs:
                    raise ValueError(f"Index '{idx}' is duplicate (or a negative index is equivalent) - indexes in Sparse Index Method must be unique.")
            real_idxs.add(real_idx)
            new_idxs.append(real_idx)
        return new_idxs

def get_method(indexes: str):
    IndexMethod = []
    idxs = []
    unique_idxs = set()
    # get indeces from string
    str_idxs = [x.strip() for x in indexes.strip().split(",")]
    for str_idx in str_idxs:
        try:
            idx = int(str_idx)
            if idx in unique_idxs:
                raise ValueError(f"'{idx}' is duplicated; indexes must be unique.")
            idxs.append(idx)
            unique_idxs.add(idx)
        except ValueError:
            raise ValueError(f"'{str_idx}' is not a valid integer index.")
    if len(idxs) == 0:
        raise ValueError(f"No indexes were listed in Sparse Index Method.")
    return (BatchIndexMethod(idxs),)