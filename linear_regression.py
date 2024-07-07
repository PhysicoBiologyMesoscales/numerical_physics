import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Iterable
from numpy.typing import NDArray


class LinearRegression_xr(LinearRegression):

    @staticmethod
    def set_input_data(
        ds: xr.Dataset, training_fields: Iterable[str], average: bool
    ) -> NDArray:
        exclude_dims = ["theta"] if average else []
        kwargs = {"type": "vector", "average": 1} if average else {"type": "vector"}
        X = np.stack(
            [
                np.concatenate(
                    [
                        ds.filter_by_attrs(name=vector_field, dir="x", **kwargs)
                        .to_dataarray()
                        .broadcast_like(ds, exclude=exclude_dims)
                        .data.flatten(),
                        ds.filter_by_attrs(name=vector_field, dir="y", **kwargs)
                        .to_dataarray()
                        .broadcast_like(ds, exclude=exclude_dims)
                        .data.flatten(),
                    ]
                )
                for vector_field in training_fields
            ],
            axis=-1,
        )
        return X

    @staticmethod
    def set_target_data(ds: xr.Dataset, target_field: str, average: bool):
        exclude_dims = ["theta"] if average else []
        kwargs = {"type": "vector", "average": 1} if average else {"type": "vector"}
        y = np.concatenate(
            [
                ds.filter_by_attrs(name=target_field, dir="x", **kwargs)
                .to_dataarray()
                .broadcast_like(ds, exclude=exclude_dims)
                .data.flatten(),
                ds.filter_by_attrs(name=target_field, dir="y", **kwargs)
                .to_dataarray()
                .broadcast_like(ds, exclude=exclude_dims)
                .data.flatten(),
            ]
        )
        return y

    def fit(
        self,
        ds: xr.Dataset,
        target_field: str = "force",
        training_fields: Iterable[str] = ["density_gradient", "polarity"],
        average: bool = True,
    ):
        X = self.set_input_data(ds, training_fields, average)
        y = self.set_target_data(ds, target_field, average)
        super().fit(X, y)

    def score(
        self,
        ds: xr.Dataset,
        target_field: str = "force",
        training_fields: Iterable[str] = ["density_gradient", "polarity"],
        average: bool = True,
        t: float = None,
    ):
        ds_ = ds.sel(t=t) if t else ds
        X = self.set_input_data(ds_, training_fields, average)
        y = self.set_target_data(ds_, target_field, average)
        return super().score(X, y)

    def predict_on_dataset(
        self,
        ds: xr.Dataset,
        data_fields: Iterable[str] = ["density_gradient", "polarity"],
        pred_field_name: str = "force",
        average: bool = True,
    ):
        Xpred = self.set_input_data(ds, data_fields, average)
        ypred = super().predict(Xpred)
        sizes = dict(ds.sizes)
        if average:
            sizes.pop("theta")
        predx, predy = ypred.reshape((2,) + tuple(sizes.values()))
        return ds.assign(
            {
                f"{pred_field_name}_predx": (
                    list(sizes.keys()),
                    predx,
                    {"name": "force_pred", "average": 1, "type": "vector", "dir": "x"},
                ),
                f"{pred_field_name}_predy": (
                    list(sizes.keys()),
                    predy,
                    {"name": "force_pred", "average": 1, "type": "vector", "dir": "y"},
                ),
            }
        )
