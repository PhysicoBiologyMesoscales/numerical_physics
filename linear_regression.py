import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Iterable
from numpy.typing import NDArray


class LinearRegression_xr(LinearRegression):

    @staticmethod
    def set_input_data(
        ds: xr.Dataset,
        training_fields: Iterable[str],
    ) -> NDArray:
        X = np.stack(
            [
                np.concatenate(
                    [ds[f"{field}x"].data.flatten(), ds[f"{field}y"].data.flatten()]
                )
                for field in training_fields
            ],
            axis=-1,
        )
        return X

    @staticmethod
    def set_target_data(ds: xr.Dataset, target_field: str):
        y = np.concatenate(
            [
                ds[f"{target_field}x"].data.flatten(),
                ds[f"{target_field}y"].data.flatten(),
            ]
        )
        return y

    def fit(
        self,
        ds: xr.Dataset,
        target_field: str = "F",
        training_fields: Iterable[str] = ["grad_rho", "p"],
    ):
        X = self.set_input_data(ds, training_fields)
        y = self.set_target_data(ds, target_field)
        super().fit(X, y)

    def score(
        self,
        ds: xr.Dataset,
        target_field: str = "F",
        training_fields: Iterable[str] = ["grad_rho", "p"],
        t: float = None,
    ):
        ds_ = ds.sel(t=t) if t else ds
        X = self.set_input_data(ds_, training_fields)
        y = self.set_target_data(ds_, target_field)
        return super().score(X, y)

    def predict_on_dataset(
        self,
        ds: xr.Dataset,
        data_fields: Iterable[str] = ["grad_rho", "p"],
        pred_field_name: str = "F",
    ):
        Xpred = self.set_input_data(ds, data_fields)
        ypred = super().predict(Xpred)
        predx, predy = ypred.reshape((2,) + tuple(ds.sizes.values()))
        return ds.assign(
            {
                f"{pred_field_name}_predx": (tuple(ds.sizes.keys()), predx),
                f"{pred_field_name}_predy": (tuple(ds.sizes.keys()), predy),
            }
        )
