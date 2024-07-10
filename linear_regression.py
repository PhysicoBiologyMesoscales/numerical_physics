import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Iterable
from numpy.typing import NDArray


class LinearRegression_xr(LinearRegression):

    def __init__(self, target_field, training_fields, poly_degree: int = 1):
        super().__init__()
        self.training_fields = None
        self.target_field = None
        self.dims = None
        self.exclude_dims = None
        self.target_field = target_field
        self.training_fields = sorted(training_fields)
        self.poly_degree = poly_degree

    def set_input_data(self, ds: xr.Dataset) -> NDArray:
        # Sorting is necessary to avoid mismatch in data order
        ds = ds.broadcast_like(ds)
        if self.dims is None or self.exclude_dims is None:
            # Compute dimensions of data
            self.dims = ds.dims
            self.exclude_dims = set(ds.dims).difference(set(self.dims))
        X = np.concatenate(
            [
                np.stack(
                    [
                        np.concatenate(
                            [
                                (
                                    ds.rho**i
                                    * ds.filter_by_attrs(
                                        name=vector_field, type="vector", dir="x"
                                    )
                                )
                                .to_dataarray()
                                .broadcast_like(ds, exclude=self.exclude_dims)
                                .data.flatten(),
                                (
                                    ds.rho**i
                                    * ds.filter_by_attrs(
                                        name=vector_field, type="vector", dir="y"
                                    )
                                )
                                .to_dataarray()
                                .broadcast_like(ds, exclude=self.exclude_dims)
                                .data.flatten(),
                            ]
                        )
                        for vector_field in self.training_fields
                    ],
                    axis=-1,
                )
                for i in range(self.poly_degree)
            ],
            axis=1,
        )

        return X

    def set_target_data(self, ds: xr.Dataset) -> NDArray:
        if self.exclude_dims is None or self.dims is None:
            raise ValueError("Dimensions not set; please set input data first")
        ds = ds.broadcast_like(ds, exclude=self.exclude_dims)
        y = np.concatenate(
            [
                ds.filter_by_attrs(name=self.target_field, type="vector", dir="x")
                .to_dataarray()
                .data.flatten(),
                ds.filter_by_attrs(name=self.target_field, type="vector", dir="y")
                .to_dataarray()
                .data.flatten(),
            ]
        )
        return y

    def fit(self, ds: xr.Dataset):
        X = self.set_input_data(ds)
        y = self.set_target_data(ds)
        super().fit(X, y)

    def score(
        self,
        ds: xr.Dataset,
        t: float = None,
    ):
        ds_ = ds.sel(t=t, method="nearest") if t else ds
        X = self.set_input_data(ds_)
        y = self.set_target_data(ds_)
        return super().score(X, y)

    def predict_on_dataset(
        self,
        ds: xr.Dataset,
    ):
        Xpred = self.set_input_data(ds)
        ypred = super().predict(Xpred)
        sizes = {dim: ds.sizes[dim] for dim in self.dims}
        predx, predy = ypred.reshape((2,) + tuple(sizes.values()))
        return ds.assign(
            {
                f"{self.target_field}_predx": (
                    list(self.dims),
                    predx,
                    {"name": "force_pred", "average": 1, "type": "vector", "dir": "x"},
                ),
                f"{self.target_field}_predy": (
                    list(self.dims),
                    predy,
                    {"name": "force_pred", "average": 1, "type": "vector", "dir": "y"},
                ),
            }
        )
