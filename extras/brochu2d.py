from pathlib import Path

import numpy as np
from numpy.random import default_rng
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from apl.apl import ActivePreferenceLearning
from apl.acquisitions import UpperConfidenceBound
from apl.posterior_approximation import ProbitDataGivenF


FILE_PATH = Path(__file__).parent.absolute()


def f(x1, x2):
    return np.clip(
        np.sin(x1)
        + x1 / 3
        + np.sin(12 * x1)
        + np.sin(x2)
        + x2 / 3
        + np.sin(12 * x2)
        - 1,
        a_min=0,
        a_max=None,
    )


def sidebar():
    st.sidebar.header("Utility Function Settings")
    x1_upper = st.sidebar.slider(
        "X1 Upper Bound", min_value=0.0, max_value=1.5, value=1.0, step=0.1
    )
    x1_lower = st.sidebar.slider(
        "X1 Lower Bound",
        min_value=-1.5,
        max_value=x1_upper,
        value=0.0,
        step=0.1,
    )
    x2_upper = st.sidebar.slider(
        "X2 Upper Bound", min_value=0.0, max_value=1.5, value=1.0, step=0.1
    )
    x2_lower = st.sidebar.slider(
        "X2 Lower Bound",
        min_value=-1.5,
        max_value=x2_upper,
        value=0.0,
        step=0.1,
    )
    total_item_size = st.sidebar.slider(
        "Item Set Size", min_value=100, max_value=1500, value=500, step=100
    )
    f_bounds = {"x1": (x1_lower, x1_upper), "x2": (x2_lower, x2_upper)}
    grid_res = st.sidebar.slider(
        "Grid Resolution for Plotting",
        min_value=20,
        max_value=100,
        step=10,
        value=30,
    )
    sigma_err = st.sidebar.slider(
        "Judgement Error Standard Deviation",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    st.sidebar.header("Algo Settings")
    kappa = st.sidebar.slider(
        "UCB Kappa (larger means more exploration)",
        min_value=0.0,
        max_value=6.0,
        value=0.0,
        step=0.05,
    )
    sigma_pred = st.sidebar.slider(
        "Assumed Judgement Error Standard Deviation",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )
    learning_max_iter = st.sidebar.slider(
        "Number of Queries", min_value=2, max_value=50, value=20, step=1
    )
    matern_kernel_length_scale = st.sidebar.selectbox(
        "Matern Kernel Length Scale", [1e-2, 1e-1, 5e-1, 1e0]
    )
    matern_kernel_nu = st.sidebar.selectbox(
        "Matern Kernel Nu", [0.5, 1.5, 2.5, np.inf]
    )
    return (
        f_bounds,
        total_item_size,
        grid_res,
        sigma_err,
        kappa,
        sigma_pred,
        learning_max_iter,
        matern_kernel_length_scale,
        matern_kernel_nu,
    )


if __name__ == "__main__":
    (
        f_bounds,
        total_item_size,
        grid_res,
        sigma_err,
        kappa,
        sigma_pred,
        learning_max_iter,
        matern_kernel_length_scale,
        matern_kernel_nu,
    ) = sidebar()
    x1_grid = np.linspace(
        start=f_bounds["x1"][0], stop=f_bounds["x1"][1], num=grid_res
    )
    x2_grid = np.linspace(
        start=f_bounds["x2"][0], stop=f_bounds["x2"][1], num=grid_res
    )
    st.title("Active Preference Learning Demo")
    st.header("Brief Explanation of the Procedure")
    st.markdown(open(FILE_PATH / "brochu2d" / "text_1.md").read())

    st.header("Borchu's 2D as Mock Utility Function")
    st.markdown(open(FILE_PATH / "brochu2d" / "text_2.md").read())
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    f_plot = f(x1_mesh, x2_mesh)
    f_plot = (f_plot - f_plot.mean()) / f_plot.std()

    fig = go.Figure(
        data=[go.Surface(z=f_plot, x=x1_mesh, y=x2_mesh, opacity=0.9)]
    )
    st.write(fig)

    st.markdown(open(FILE_PATH / "brochu2d" / "text_3.md").read())

    rng = default_rng(0)

    x1 = rng.uniform(
        low=f_bounds["x1"][0], high=f_bounds["x1"][1], size=total_item_size
    )
    x2 = rng.uniform(
        low=f_bounds["x2"][0], high=f_bounds["x2"][1], size=total_item_size
    )
    f_train = f(x1, x2)
    f_train_mean, f_train_std = f_train.mean(), f_train.std()
    f_train = (f_train - f_train_mean) / f_train_std
    # u_train = f_train + rng.normal(scale=sigma_err, size=total_item_size)

    X = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    explored_item_idx = np.array([], dtype=np.int32)
    query_item_idx = np.arange(total_item_size, dtype=np.int32)

    kernel = Matern(
        length_scale=matern_kernel_length_scale, nu=matern_kernel_nu
    ) + WhiteKernel(noise_level=sigma_pred)
    apl = ActivePreferenceLearning(
        kernel=kernel,
        acquisition=UpperConfidenceBound(kappa=kappa),
        loglikelihood=ProbitDataGivenF(sigma_pred),
        random_state=3,
    )

    # ------------------- RUN ALGO ----------------------- #

    opt1_idx, opt2_idx, explored_item_idx, query_item_idx, mu = apl.query(
        X,
        explored_item_idx,
        query_item_idx,
        mu=None,
        pair_selections=None,
    )

    pair_selections = np.empty((1, 2), dtype=np.int16)
    pref_idx = np.argwhere(explored_item_idx == opt1_idx).item()
    sugg_idx = np.argwhere(explored_item_idx == opt2_idx).item()

    algo_trace = []

    if f_train[opt1_idx] + rng.normal(scale=sigma_err) > f_train[
        opt2_idx
    ] + rng.normal(
        scale=sigma_err
    ):  # keep preference
        algo_trace.append(f_train[opt1_idx])
        pair_selections[0] = [pref_idx, sugg_idx]
    else:  # replace preference with suggestions
        algo_trace.append(f_train[opt2_idx])
        pair_selections[0] = [sugg_idx, pref_idx]

    for it in range(learning_max_iter):
        opt1_idx, opt2_idx, explored_item_idx, query_item_idx, mu = apl.query(
            X,
            explored_item_idx,
            query_item_idx,
            mu=mu,
            pair_selections=pair_selections,
        )
        pref_idx = np.argwhere(explored_item_idx == opt1_idx).item()
        sugg_idx = np.argwhere(explored_item_idx == opt2_idx).item()
        if f_train[opt1_idx] + rng.normal(scale=sigma_err) > f_train[
            opt2_idx
        ] + rng.normal(
            scale=sigma_err
        ):  # keep preference
            algo_trace.append(f_train[opt1_idx])
            new_selection = np.array([pref_idx, sugg_idx])
        else:  # replace preference with suggestions
            algo_trace.append(f_train[opt2_idx])
            new_selection = np.array([sugg_idx, pref_idx])

        pair_selections = np.vstack((pair_selections, new_selection))

    X_train = X[explored_item_idx]

    # ------------------- RUN RANDOM ----------------------- #
    random_picks = np.random.choice(
        np.arange(X.shape[0]), size=len(algo_trace)
    )
    random_pick_trace = [f_train[random_picks[0].item()]]
    current_best = random_picks[0].item()
    for contender in random_picks[1:]:
        if f_train[current_best] + rng.normal(scale=sigma_err) > f_train[
            contender.item()
        ] + rng.normal(
            scale=sigma_err
        ):  # keep preference
            random_pick_trace.append(f_train[current_best])
        else:  # replace preference with suggestions
            random_pick_trace.append(f_train[contender.item()])
            current_best = contender.item()

    # ------------------- PRESENT RESULTS ----------------------- #
    st.header("Algo In Action")
    st.markdown(open(FILE_PATH / "brochu2d" / "text_4.md").read())
    fig = go.Figure()
    fig.add_trace(go.Surface(z=f_plot, x=x1_mesh, y=x2_mesh, opacity=0.5))
    f_train_plot = (
        f(X_train[:, 0], X_train[:, 1]) - f_train_mean
    ) / f_train_std + 3e-2
    fig.add_trace(
        go.Scatter3d(
            z=f_train_plot,
            x=X_train[:, 0],
            y=X_train[:, 1],
            mode="markers",
            marker=dict(color="black", size=10, symbol="cross"),
        )
    )

    st.write(fig)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(len(algo_trace)), y=algo_trace, name="APL")
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(algo_trace)),
            y=np.maximum.accumulate(random_pick_trace),
            name="Random Picks",
        )
    )
    st.write(fig)

    st.header("Things to Try")
    st.markdown(open(FILE_PATH / "brochu2d" / "text_5.md").read())
