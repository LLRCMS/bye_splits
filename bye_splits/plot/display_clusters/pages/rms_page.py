import os
import sys
from dash import dcc, html, Input, Output, callback, ctx
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import yaml

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, cl_helpers

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

with open(params.CfgPaths["cluster_app"], "r") as afile:
    cfgprod = yaml.safe_load(afile)

pile_up_dir = "PU0" if not cfgprod["clusterSize"]["pileUp"] else "PU200"

if cfgprod["dirs"]["local"]:
    data_dir = cfgprod["dirs"]["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, "data/")

input_files = cl_helpers.get_output_files(cfgprod)

with open(params.CfgPaths["cluster_app"], "r") as afile:
    cfgprod = yaml.safe_load(afile)


def rms(data):
    return np.sqrt(np.mean(np.square(data)))


def effrms(data, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    assert data.shape == (data.shape[0],)
    new_series = data.dropna()
    x = np.sort(new_series, kind="mergesort")
    m = int(c * len(x)) + 1
    # out = [np.min(x[m:] - x[:-m]) / 2.0]
    out = np.min(x[m:] - x[:-m]) / 2.0

    return out


def get_rms(coef, eta_range, normby, rms_dict=None, rms_eff_dict=None, binned_rms=True):
    dfs_by_particle = cl_helpers.get_dfs(input_files, coef)

    # Older files were written using singular keys, this ensures standardization to the new format
    dict_keys = ["photons", "pions"]
    dfs_by_particle = dict(zip(dict_keys, dfs_by_particle.values()))

    phot_df, pion_df = dfs_by_particle["photons"], dfs_by_particle["pions"]

    if normby == "Energy":
        phot_df["normed_energies"] = phot_df["en"] / phot_df["gen_en"]
        pion_df["normed_energies"] = pion_df["en"] / pion_df["gen_en"]
    else:
        phot_df["pt"] = (
            phot_df["en"] / np.cosh(phot_df["eta"])
            if "pt" not in phot_df.keys()
            else phot_df["pt"]
        )
        pion_df["pt"] = (
            pion_df["en"] / np.cosh(pion_df["eta"])
            if "pt" not in pion_df.keys()
            else pion_df["pt"]
        )

        phot_df["normed_energies"] = phot_df["pt"] / phot_df["genpart_pt"]
        pion_df["normed_energies"] = pion_df["pt"] / pion_df["genpart_pt"]

    phot_df = phot_df[
        (phot_df.gen_eta > eta_range[0]) & (phot_df.gen_eta < eta_range[1])
    ]
    pion_df = pion_df[
        (pion_df.gen_eta > eta_range[0]) & (pion_df.gen_eta < eta_range[1])
    ]

    phot_mean_en = phot_df["normed_energies"].mean()
    pion_mean_en = pion_df["normed_energies"].mean()

    phot_rms = phot_df["normed_energies"].std() / phot_mean_en
    phot_eff_rms = effrms(phot_df["normed_energies"]) / phot_mean_en

    pion_rms = pion_df["normed_energies"].std() / pion_mean_en
    pion_eff_rms = effrms(pion_df["normed_energies"]) / pion_mean_en

    if binned_rms:
        return (
            phot_df,
            pion_df,
            {"Photon": phot_rms, "Pion": pion_rms},
            {"Photon": phot_eff_rms, "Pion": pion_eff_rms},
        )
    else:
        rms_dict["Photon"] = np.append(rms_dict["Photon"], phot_rms)
        rms_eff_dict["Photon"] = np.append(rms_eff_dict["Photon"], phot_eff_rms)

        rms_dict["Pion"] = np.append(rms_dict["Pion"], pion_rms)
        rms_eff_dict["Pion"] = np.append(rms_eff_dict["Pion"], pion_eff_rms)


# Dash page setup
##############################################################################################################################

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="RMS", name="RMS")

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Interactive Normal Distribution",
                    style={"fontSize": 40, "textAlign": "center"},
                )
            ]
        ),
        html.Hr(),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", disabled=True)]),
        html.Hr(),
        dcc.Graph(id="histograms-x-graph", mathjax=True),
        html.P("Coef:"),
        dcc.Slider(id="coef", min=0.0, max=0.05, value=0.001, marks=marks),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.4, max=2.7, step=0.1, value=[1.6, 2.7]),
        html.P("Normalization:"),
        dcc.Dropdown(["Energy", "PT"], "PT", id="normby"),
        html.Hr(),
        dbc.Row(
            [
                dcc.Markdown(
                    r"Gaussianity := $\frac{|RMS-RMS_{Eff}|}{RMS}$",
                    mathjax=True,
                    style={"fontSize": 30, "textAlign": "center"},
                )
            ]
        ),
        html.Hr(),
        html.Div(id="my_table"),
        html.Hr(),
        dbc.Row(
            [dcc.Markdown("Resolution", style={"fontSize": 30, "textAlign": "center"})]
        ),
        dcc.Graph(id="global-rms-graph", mathjax=True),
    ]
)


@callback(
    Output("histograms-x-graph", "figure"),
    Output("my_table", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
)

##############################################################################################################################


def plot_dists(coef, eta_range, normby, pileup):
    button_clicked = ctx.triggered_id
    pile_up = True if button_clicked == "pileup" else False

    phot_df, pion_df, rms, eff_rms = get_rms(coef, eta_range, normby)

    phot_rms, pion_rms = rms["Photon"], rms["Pion"]
    phot_eff_rms, pion_eff_rms = eff_rms["Photon"], eff_rms["Pion"]

    phot_gaus_diff = np.abs(phot_eff_rms - phot_rms) / phot_rms
    pion_gaus_diff = np.abs(pion_eff_rms - pion_rms) / pion_rms

    pion_gaus_str = format(pion_gaus_diff, ".3f")
    phot_gaus_str = format(phot_gaus_diff, ".3f")

    pion_rms_str = format(pion_rms, ".3f")
    phot_rms_str = format(phot_rms, ".3f")

    pion_eff_rms_str = format(pion_eff_rms, ".3f")
    phot_eff_rms_str = format(phot_eff_rms, ".3f")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pion_df["normed_energies"], nbinsx=100, autobinx=False, name="Pion"
        )
    )
    fig.add_trace(
        go.Histogram(
            x=phot_df["normed_energies"], nbinsx=100, autobinx=False, name="Photon"
        )
    )

    if normby == "Energy":
        x_title = r"$\Huge{\frac{E_{Cl}}{E_{Gen}}}$"
    else:
        x_title = r"$\Huge{\frac{{p_T}^{Cl}}{{p_T}^{Gen}}}$"

    fig.update_layout(
        barmode="overlay",
        title_text="Normalized Cluster {}".format(normby),
        xaxis_title=x_title,
        yaxis_title_text=r"$\Large{Events}$",
    )

    fig.update_traces(opacity=0.5)

    my_vals = {
        "Photon": {
            "RMS": phot_rms_str,
            "Effective RMS": phot_eff_rms_str,
            "Gaussianity": phot_gaus_str,
        },
        "Pion": {
            "RMS": pion_rms_str,
            "Effective RMS": pion_eff_rms_str,
            "Gaussianity": pion_gaus_str,
        },
    }

    val_df = pd.DataFrame(my_vals).reset_index()
    val_df = val_df.rename(columns={"index": ""})

    return fig, dbc.Table.from_dataframe(val_df)


def write_rms_file(coefs, eta, norm, filename):
    rms_by_part = {"Photon": [], "Pion": []}

    rms_eff_by_part = {"Photon": [], "Pion": []}

    for coef in coefs[1:]:
        get_rms(coef, eta, norm, rms_by_part, rms_eff_by_part, binned_rms=False)
    with pd.HDFStore(filename, "w") as glob_rms_file:
        glob_rms_file.put("RMS", pd.DataFrame.from_dict(rms_by_part))
        glob_rms_file.put("Eff_RMS", pd.DataFrame.from_dict(rms_eff_by_part))

    return rms_by_part, rms_eff_by_part


@callback(
    Output("global-rms-graph", "figure"),
    Input("eta_range", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
)
def glob_rms(eta_range, normby, pileup, file="rms_and_eff.hdf5"):
    button_clicked = ctx.triggered_id
    pile_up = True if button_clicked == "pileup" else False

    coefs = cl_helpers.get_keys(input_files)

    filename = "{}_eta_{}_{}_{}".format(
        normby, str(eta_range[0]), str(eta_range[1]), file
    )
    filename_user = "{}{}/{}".format(data_dir, pile_up_dir, filename)
    filename_iehle = "{}{}{}/{}".format(
        cfgprod["dirs"]["ehleDir"], cfgprod["dirs"]["dataFolder"], pile_up_dir, filename
    )

    if os.path.exists(filename_user):
        with pd.HDFStore(filename_user, "r") as glob_rms_file:
            rms_by_part, rms_eff_by_part = glob_rms_file["/RMS"].to_dict(
                orient="list"
            ), glob_rms_file["/Eff_RMS"].to_dict(orient="list")
    elif os.path.exists(filename_iehle):
        with pd.HDFStore(filename_iehle, "r") as glob_rms_file:
            rms_by_part, rms_eff_by_part = glob_rms_file["/RMS"].to_dict(
                orient="list"
            ), glob_rms_file["/Eff_RMS"].to_dict(orient="list")
    else:
        rms_by_part, rms_eff_by_part = write_rms_file(
            coefs, eta_range, normby, filename_user
        )

    nice_coefs = np.linspace(0.0, 0.05, 50)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    for var in ["RMS", "Eff-RMS"]:
        for particle, color in zip(["Photon", "Pion"], ["purple", "red"]):
            fig.add_trace(
                go.Scatter(
                    x=nice_coefs,
                    y=rms_by_part[particle]
                    if var == "RMS"
                    else rms_eff_by_part[particle],
                    name=var,
                    line_color=color,
                    mode="lines" if var == "RMS" else "markers",
                ),
                row=1,
                col=1 if particle == "Photon" else 2,
            )

    fig.update_xaxes(title_text="Radius (Coefficient)")

    fig.update_layout(
        title_text="Resolution in {}".format(normby), yaxis_title_text="(Effective) RMS"
    )

    return fig
