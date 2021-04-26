from typing import List
from typing_extensions import Protocol
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    pass


class SupportsSelector(Protocol):
    new_cols: List[str]
    
    def get_new_cols(self) -> List[str]:
        return self.new_cols