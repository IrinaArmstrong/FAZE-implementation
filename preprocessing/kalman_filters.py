# Basic
import numpy as np
from typing import Union

import logging_handler

logger = logging_handler.get_logger(__name__)


class KalmanFilter1D:
    """
    Implements a Kalman filter for bounding box location filtering.
    Using FAZE authors's implementation of filter,
    which is working with complex number measurement's types.
    """

    def __init__(self, dim_x: int = 100,
                 P_: float = 1.0,
                 R_: float = 1e-06,
                 Q_: float = 1e-5, init_x: float = 0.0):
        """
        Parameters
        ----------
        defaut: dim_z = 1 as filter is 1-dim
        :param dim_x: int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.
            This is used to set the default size of cov matrixes F, P, Q.
        Note: In this implementation do not equals to number of measurement inputs.
        :param R_: initial value for main diagonal of measurement noise matrix R [dim_z, dim_z].
        :param Q_: initial value for main diagonal of process noise matrix [dim_x, dim_x],
                  usually initalized as process variance.
        :param P_: initial value for main diagonal of state covariance matrix P [dim_x, dim_x]
        """
        if dim_x < 1:
            logger.error(f'Number of state variables `dim_x` must be 1 or greater!')
            raise ValueError('Number of state variables `dim_x` must be 1 or greater!')
        self._dim_x = dim_x

        self.x = np.zeros(dim_x, dtype=np.complex128)  # a posteri estimate of x
        self._z = np.zeros(dim_x, dtype=np.complex128)  # a priori estimate of x
        self._Q = Q_  # Process noise matrix (dim_x, dim_x), process uncertainty
        self._P = np.zeros(dim_x, dtype=np.complex128)  # a posteri error estimate (current state covariance matrix)
        self._R = R_  # Measurement noise matrix, estimate of measurement variance (state uncertainty)

        # Gain and residual are computed during the innovation step
        self._y = np.zeros(dim_x, dtype=np.complex128)  # a priori error estimate
        self._K = np.zeros(dim_x, dtype=np.complex128)  # gain or blending factor (Kalman Gain)

        # Intial guesses
        self.x[0] = init_x
        self._P[0] = P_
        self.__k = 1  # current prediction index

    def update(self, z_k: Union[float, np.complex128]):
        k = self.__k % self._dim_x
        km = (self.__k - 1) % self._dim_x
        self._z[k] = self.x[km]
        self._y[k] = self._P[km] + self._Q

        # measurement update
        self._K[k] = self._y[k] / (self._y[k] + self._R)
        self.x[k] = self._z[k] + self._K[k] * (z_k - self._z[k])
        self._P[k] = (1 - self._K[k]) * self._y[k]
        self.__k = self.__k + 1
        return self.x[k]


if __name__ == "__main__":
    import plotly.graph_objects as go

    # Data
    n_samples = 50
    x_axis = np.linspace(-np.pi, np.pi, num=n_samples)
    x_true = np.sin(x_axis)
    # Noise component
    noise = np.random.normal(loc=0.0, scale=0.1, size=n_samples)
    x_noised = x_true + noise

    ff = KalmanFilter1D(dim_x=10, P_=1.0, R_=1e-06, Q_=1e-5)
    filtered_x = []
    state_cov_history = []

    for i, x_sample in enumerate(x_noised):
        x_f = np.real(ff.update(x_sample))  # Этап предсказания + коррекции
        print(f"\n#{i}: {x_sample} -> {x_f}")
        filtered_x.append(x_f)
        state_cov_history.append(ff._P)

    filtered_x = np.array(filtered_x)
    state_cov_history = np.array(state_cov_history)

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=x_true, mode='lines', name='True values'))
    fig.add_trace(go.Scatter(x=x_axis, y=x_noised, mode='lines', name='Noised true values'))
    fig.add_trace(go.Scatter(x=x_axis, y=filtered_x, mode='lines', name='Filtered values'))
    fig.show()
