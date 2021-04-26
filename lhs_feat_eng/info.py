from tkinter import S
from typing import Any, Callable, Sequence
from typing_extensions import Protocol
import pandas as pd
import numpy as np

from . import base


class GroupFunc(Protocol):
    
    def __call__(self, s: pd.Series) -> pd.Series:
        ...


class IndicatorFunc(Protocol):
    
    def __call__(self, prices: pd.Series, byseq: pd.Series, bins: pd.Series) -> pd.Series:
        ...


def _get_bin_cap(s: pd.Series, n: int = 50) -> float:
    return s.sum() / len(s) * n 


def equal_weighted_group(s: pd.Series,
                         get_bin_cap: Callable[[pd.Series], float]) -> pd.Series:
    
    series = pd.Series(s)
    
    bin_cap = get_bin_cap(series)
    bins = np.array(series.cumsum() / bin_cap, dtype='int')
    return pd.Series(bins, index=series.index)

def get_vpin(prices: pd.Series, byseq: pd.Series, bins: pd.Series) -> pd.Series:
    pseries = pd.Series(prices)
    vseries = pd.Series(byseq)
    
    diff = pseries.diff()
    inc = diff > 0
    dec = diff <= 0
    bar_buy = vseries * inc
    bar_sell = vseries * dec
    
    vol_bin = vseries.groupby(bins).sum()
    vol_buy = bar_buy.groupby(bins).sum()
    vol_sell = bar_sell.groupby(bins).sum()
    bin_bars = vseries.groupby(bins).agg(lambda x: len(x))
    vpin = (vol_buy - vol_sell).abs() / vol_bin / bin_bars
    return vpin


def resample_by_bar(
    prices: Sequence[float],
    byseq: Sequence[float],
    group_func: GroupFunc,
    indicator_func: IndicatorFunc,
) -> pd.Series:
    pseries = pd.Series(prices)
    vseries = pd.Series(byseq)
    bins = group_func(vseries)

    return indicator_func(pseries, vseries, bins)
    
    
def get_simple_vpin(
    prices: Sequence[float],
    byseq: Sequence[float],
) -> pd.Series:
    return resample_by_bar(prices=prices,
                           byseq=byseq,
                           group_func=lambda s: equal_weighted_group(s, get_bin_cap=_get_bin_cap),
                           indicator_func=get_vpin)