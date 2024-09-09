import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray


class LinearRegression_xr(LinearRegression):

    def __init__(self, target_field):
        super().__init__()
        self.training_fields = None
        self.target_field = None
        self.dims = None
        self.training_fields = None
        self.target_field = target_field

    def set_input_data(self, ds: xr.Dataset) -> NDArray:

        if self.dims is None:
            # Compute dimensions of data
            self.dims = list(ds.sizes)

        def list_fields(ds: xr.Dataset, **kwargs):
            list_fields = []
            for field in ds.filter_by_attrs(**kwargs):
                field_name = ds[field].attrs["name"]
                if field_name in list_fields:
                    continue
                list_fields.append(field_name)
            return list_fields

        self.training_fields = list_fields(ds, training=1)

        X = np.stack(
            [
                np.concatenate(
                    [
                        (ds[f"{vector_field}_x"]).data.flatten(),
                        (ds[f"{vector_field}_y"]).data.flatten(),
                    ]
                )
                for vector_field in self.training_fields
            ],
            axis=-1,
        )
        return X

    def set_target_data(self, ds: xr.Dataset) -> NDArray:
        y = np.concatenate(
            [
                ds[f"{self.target_field}_x"].data.flatten(),
                ds[f"{self.target_field}_y"].data.flatten(),
            ]
        )
        return y

    def fit(self, ds: xr.Dataset):
        X = self.set_input_data(ds)
        y = self.set_target_data(ds)
        super().fit(X, y)

    def score_on_dataset(
        self,
        ds: xr.Dataset,
        t=None,
    ):
        match t:
            case float() | int():
                ds_ = ds.sel(t=t, method="nearest")
            case slice():
                ds_ = ds.sel(t=t)
            case _:
                ds_ = ds
        X = self.set_input_data(ds_)
        y = self.set_target_data(ds_)
        return super().score(X, y)

    def predict_on_dataset(
        self,
        ds: xr.Dataset,
    ):
        Xpred = self.set_input_data(ds)
        ypred = super().predict(Xpred)
        predx, predy = ypred.reshape((2,) + tuple(ds.sizes[dim] for dim in self.dims))
        return ds.assign(
            {
                f"{self.target_field}_pred_x": (
                    self.dims,
                    predx,
                    {
                        "name": f"{self.target_field}_pred",
                        "type": "vector",
                        "dir": "x",
                    },
                ),
                f"{self.target_field}_pred_y": (
                    self.dims,
                    predy,
                    {
                        "name": f"{self.target_field}_pred",
                        "type": "vector",
                        "dir": "y",
                    },
                ),
            }
        )
