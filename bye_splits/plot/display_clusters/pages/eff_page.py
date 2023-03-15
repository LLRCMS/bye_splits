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

parser = argparse.ArgumentParser(description="Clustering standalone step.")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

with open(params.CfgPath, "r") as afile:
    cfg = yaml.safe_load(afile)

pile_up_dir = "PU0" if not cfg["clusterSize"]["pileUp"] else "PU200"

if cfg["clusterStudies"]["local"]:
    data_dir = cfgprod["clusterStudies"]["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, "data/")

input_files = cl_helpers.get_output_files(cfg)


def binned_effs(df, norm, perc=0.1):
    """Takes a dataframe 'df' with a column 'norm' to normalize by, and returns
    1) a binned matching efficiency list
    2) a binned list corresponding to 'norm'
    where the binning is done by percentage 'perc' of the size of the 'norm' column"""
    eff_list = [0]
    en_list = [0]
    en_bin_size = perc * (df[norm].max() - df[norm].min())
    if perc < 1.0:
        current_en = 0
        for i in range(1, 101):
            match_column = df.loc[
                df[norm].between(current_en, (i) * en_bin_size, "left"), "matches"
            ]
            if not match_column.empty:
                try:
                    eff = float(match_column.value_counts(normalize=True))
                except TypeError:
                    eff = match_column.value_counts(normalize=True)[True]
                eff_list.append(eff)
                current_en += en_bin_size
                en_list.append(current_en)
    else:
        match_column = df.loc[df[norm].between(0, en_bin_size, "left"), "matches"]
        if not match_column.empty:
            try:
                eff = float(match_column.value_counts(normalize=True))
            except TypeError:
                eff = match_column.value_counts(normalize=True)[True]
            eff_list = eff
    return eff_list, en_list


# Dash page setup
##############################################################################################################################

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="Efficiency", name="Efficiency")

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Reconstruction Efficiency",
                    style={"fontSize": 30, "textAlign": "center"},
                )
            ]
        ),
        html.Hr(),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", disabled=True)]),
        html.Hr(),
        dcc.Graph(id="eff-graph", mathjax=True),
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
                    "Global Efficiencies", style={"fontSize": 30, "textAlign": "center"}
                )
            ]
        ),
        html.Div(id="glob-effs"),
        html.Hr(),
        dbc.Row(
            [
                dcc.Markdown(
                    "Efficiencies By Coefficent",
                    style={"fontSize": 30, "textAlign": "center"},
                )
            ]
        ),
        dcc.Graph(id="glob-eff-graph", mathjax=True),
    ]
)


# Callback function for display_color() which displays binned efficiency/energy graphs
@callback(
    Output("eff-graph", "figure"),
    Output("glob-effs", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
)

##############################################################################################################################


def display_color(coef, eta_range, normby, pileup):
    button_clicked = ctx.triggered_id
    pile_up = True if button_clicked == "pileup" else False

    df_by_particle = cl_helpers.get_dfs(input_files, coef)
    phot_df = df_by_particle["photons"]
    pion_df = df_by_particle["pions"]

    phot_df = phot_df[
        (phot_df.gen_eta > eta_range[0]) & (phot_df.gen_eta < eta_range[1])
    ]
    pion_df = pion_df[
        (pion_df.gen_eta > eta_range[0]) & (pion_df.gen_eta < eta_range[1])
    ]

    # Bin energy data into n% chunks to check eff/energy (10% is the default)
    if normby == "Energy":
        phot_effs, phot_x = binned_effs(phot_df, "gen_en")
        pion_effs, pion_x = binned_effs(pion_df, "gen_en")
    else:
        phot_effs, phot_x = binned_effs(phot_df, "genpart_pt")
        pion_effs, pion_x = binned_effs(pion_df, "genpart_pt")

    glob_effs = pd.DataFrame(
        {"photons": np.mean(phot_effs[1:]), "pions": np.mean(pion_effs[1:])}, index=[0]
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    fig.add_trace(go.Scatter(x=phot_x, y=phot_effs, name="photons"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pion_x, y=pion_effs, name="pions"), row=1, col=2)

    fig.update_xaxes(title_text="{} (GeV)".format(normby))

    fig.update_yaxes(type="log")

    fig.update_layout(
        title_text="Efficiency/{}".format(normby),
        yaxis_title_text=r"$Eff (\frac{N_{Cl}}{N_{Gen}})$",
    )

    return fig, dbc.Table.from_dataframe(glob_effs)


def write_eff_file(norm, coefs, eta, file):
    # Note that a cluster having radius (coef) zero also has zero efficiency, so we initialize as such and calculate for the following coefs (i.e. starting with coefs[1])
    effs_dict = {"photons": [0.0], "pions": [0.0]}
    for coef in coefs[1:]:
        dfs_by_particle = cl_helpers.get_dfs(input_files, coef)
        phot_df = dfs_by_particle["photons"]
        pion_df = dfs_by_particle["pions"]

        phot_df = phot_df[(phot_df.gen_eta > eta[0]) & (phot_df.gen_eta < eta[1])]
        pion_df = pion_df[(pion_df.gen_eta > eta[0]) & (pion_df.gen_eta < eta[1])]

        binned_var = "gen_en" if norm == "Energy" else "genpart_pt"
        phot_eff, _ = binned_effs(phot_df, binned_var, 1.0)
        pion_eff, _ = binned_effs(pion_df, binned_var, 1.0)

        effs_dict["photons"] = np.append(effs_dict["photons"], phot_eff)
        effs_dict["pions"] = np.append(effs_dict["pions"], pion_eff)

    with pd.HDFStore(file, "w") as glob_eff_file:
        glob_eff_file.put("Eff", pd.DataFrame.from_dict(effs_dict))

    return effs_dict


# Callback function for global_effs() which displays global efficiency as a function of the coefficent/radius
@callback(
    Output("glob-eff-graph", "figure"),
    Input("eta_range", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
)
def global_effs(eta_range, normby, pileup, file="global_eff.hdf5"):
    button_clicked = ctx.triggered_id
    pile_up = True if button_clicked == "pileup" else False

    coefs = cl_helpers.get_keys(input_files)

    filename = "{}_eta_{}_{}_{}".format(normby, eta_range[0], eta_range[1], file)
    filename_user = "{}{}/{}".format(data_dir, pile_up_dir, filename)
    filename_iehle = "{}{}{}/{}".format(
        cfg["clusterStudies"]["ehleDir"], cfg["clusterStudies"]["dataFolder"], pile_up_dir, filename
    )

    if os.path.exists(filename_user):
        with pd.HDFStore(filename_user, "r") as glob_eff_file:
            effs_by_coef = glob_eff_file["/Eff"].to_dict(orient="list")
    elif os.path.exists(filename_iehle):
        with pd.HDFStore(filename_iehle, "r") as glob_eff_file:
            effs_by_coef = glob_eff_file["/Eff"].to_dict(orient="list")
    else:
        effs_by_coef = write_eff_file(normby, coefs, eta_range, filename_user)

    # Old convention created the DFs and dicts with the keys 'Photon'/'Pion' instead of 'photons'/'pions'
    dict_keys = ["photons", "pions"]
    effs_by_coef = dict(zip(dict_keys, effs_by_coef.values()))

    coefs = np.linspace(0.0, 0.05, 50)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    fig.add_trace(
        go.Scatter(x=coefs, y=effs_by_coef["photons"], name="photons"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=coefs, y=effs_by_coef["pions"], name="pions"), row=1, col=2
    )

    fig.update_xaxes(title_text="Radius (Coefficient)")

    # Range [a,b] is defined by [10^a, 10^b], hence passing to log
    fig.update_yaxes(type="log", range=[np.log10(0.97), np.log10(1.001)])

    fig.update_layout(
        title_text="Efficiency/Radius",
        yaxis_title_text=r"$Eff (\frac{N_{Cl}}{N_{Gen}})$",
    )

    return fig
