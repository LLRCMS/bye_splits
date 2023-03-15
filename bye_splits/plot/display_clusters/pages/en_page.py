import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import common

import re
import numpy as np
import pandas as pd
import yaml

from dash import Dash, dcc, html, Input, Output, callback, ctx
import dash

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
import bye_splits
from bye_splits.utils import params, parsing, cl_helpers

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

# Old Naming Conventions used different column names in dataframes
column_matching = {
    "etanew": "eta",
    "phinew": "phi",
    "genpart_exphi": "gen_phi",
    "genpart_exeta": "gen_eta",
    "genpart_energy": "gen_en",
}

with open(params.CfgPaths["cluster_app"], "r") as afile:
    cl_cfg = yaml.safe_load(afile)

pile_up_dir = "PU0" if not cl_cfg["clusterSize"]["pileUp"] else "PU200"

if cl_cfg["dirs"]["local"]:
    data_dir = cl_cfg["dirs"]["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, cl_cfg["dirs"]["dataFolder"])

input_files = cl_helpers.get_output_files(cl_cfg)

dash.register_page(__name__, title="Energy", name="Energy")

layout = html.Div(
    [
        html.H4("Normalized Cluster Energy"),
        html.Hr(),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", disabled=True)]),
        html.Hr(),
        dcc.Graph(id="cl-en-graph", mathjax=True),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.4, max=2.7, step=0.1, value=[1.6, 2.7]),
        html.P("Normalization:"),
        dcc.Dropdown(["Energy", "PT"], "PT", id="normby"),
    ]
)


def fill_dict_w_mean_norm(key, coef, eta, df, norm, out_dict, df_needs_norm=False):
    """The df_needs_norm bool handles legacy code which calculated the norm in this file,
    while the updated version of the cluster size study calculates it directly."""
    if df_needs_norm:
        if norm == "PT":
            df["pt"] = (
                df["en"] / np.cosh(df["eta"]) if "pt" not in df.keys() else df["pt"]
            )
            df["normed_energies"] = (
                df["pt"] / df["genpart_pt"]
                if "genpart_pt" in df.keys()
                else df["pt"] / df["gen_pt"]
            )
        else:
            df["normed_energies"] = df["en"] / df["gen_en"]

        df = df[(df.gen_eta > eta[0]) & (df.gen_eta < eta[1])]

        mean_energy = df["normed_energies"].mean()
    else:
        df = df[(df.gen_eta > eta[0]) & (df.gen_eta < eta[1])]
        if norm == "PT":
            mean_energy = df["pt_norm"].mean()
        else:
            mean_energy = df["en_norm"].mean()

    out_dict[key] = np.append(out_dict[key], mean_energy)


def write_plot_file(input_files, norm, eta, outfile, pars=vars(FLAGS)):
    normed_energies = {}
    for key in input_files.keys():
        # Initialize at 0 since we only consider coefs[1:] (coefs[0] is an empty dataframe)
        if len(input_files[key]) > 0:
            normed_energies[key] = [0.0]

    for key in input_files.keys():
        if len(input_files[key]) == 0:
            continue
        elif len(input_files[key]) == 1:
            with pd.HDFStore(input_files[key][0], "r") as File:
                coef_strs = File.keys()
                for coef in coef_strs[1:]:
                    if File[coef].index.name != "event":
                        df = File[coef].set_index("event")
                    try:
                        df = df.drop(columns=["matches", "en_max"])
                        df.rename(column_matching, axis=1, inplace=True)
                    except KeyError:
                        pass
                    fill_dict_w_mean_norm(
                        key, coef, eta, df, norm, normed_energies, df_needs_norm=True
                    )
        else:
            file_list = [pd.HDFStore(val, "r") for val in input_files[key]]
            coef_strs = file_list[0].keys()
            for coef in coef_strs[1:]:
                df_list = [file_list[i][coef] for i in range(len(file_list))]
                full_df = pd.concat(df_list)
                full_df = (
                    full_df.set_index("event")
                    if not full_df.index.name == "event"
                    else full_df
                )
                try:
                    full_df = full_df.drop(columns=["matches", "en_max"])
                    full_df.rename(column_matching, axis=1, inplace=True)
                except KeyError:
                    pass
                fill_dict_w_mean_norm(
                    key, coef, eta, full_df, norm, normed_energies, df_needs_norm=True
                )

            for file in file_list:
                file.close()

    with pd.HDFStore(outfile, "w") as PlotFile:
        normed_df = pd.DataFrame.from_dict(normed_energies)
        PlotFile.put("Normed_Dist", normed_df)

    return normed_energies


@callback(
    Output("cl-en-graph", "figure"),
    Input("normby", "value"),
    Input("eta_range", "value"),
    Input("pileup", "n_clicks"),
)
def plot_norm(
    normby, eta_range, pileup, init_files=input_files, plot_file="normed_distribution"
):
    button_clicked = ctx.triggered_id
    pile_up = True if button_clicked == "pileup" else False
    global y_axis_title

    if normby == "Energy":
        y_axis_title = r"$\huge{\frac{\bar{E_{Cl}}}{\bar{E}_{Gen}}}$"
    elif normby == "PT":
        y_axis_title = r"$\huge{\frac{\bar{p_T}^{Cl}}{\bar{p_T}^{Gen}}}$"
    else:
        y_axis_title = r"$\huge{\frac{E_{Cl}}{E_{Max}}}$"

    plot_filename = "{}_{}_eta_{}_{}.hdf5".format(
        normby, plot_file, eta_range[0], eta_range[1]
    )
    plot_filename_user = "{}{}/{}".format(data_dir, pile_up_dir, plot_filename)
    plot_filename_iehle = "{}{}{}/new_{}".format(
        cl_cfg["dirs"]["ehleDir"],
        cl_cfg["dirs"]["dataFolder"],
        pile_up_dir,
        plot_filename,
    )

    if os.path.exists(plot_filename_user):
        with pd.HDFStore(plot_filename_user, "r") as PlotFile:
            normed_dist = PlotFile["/Normed_Dist"].to_dict(orient="list")
    elif os.path.exists(plot_filename_iehle):
        with pd.HDFStore(plot_filename_iehle, "r") as PlotFile:
            normed_dist = PlotFile["/Normed_Dist"].to_dict(orient="list")
    else:
        normed_dist = write_plot_file(init_files, normby, eta_range, plot_filename_user)

    # Older files were written using singular keys, this ensures standardization to the new format
    dict_keys = ["photons", "pions"]
    normed_dist = dict(zip(dict_keys, normed_dist.values()))

    start, end, tot = cl_cfg["clusterSize"]["coeffs"]
    coefs = np.linspace(start, end, tot)

    coef_labels = [round(coef, 3) for coef in coefs]
    coef_labels = coef_labels[0::5]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    en_phot, en_pion = normed_dist["photons"], normed_dist["pions"]

    fig.add_trace(go.Scatter(x=coefs, y=en_phot, name="Photon"), row=1, col=1)
    fig.add_trace(go.Scatter(x=coefs, y=en_pion, name="Pion"), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="green")

    fig.update_xaxes(title_text="Radius (Coeff)")

    fig.update_layout(
        title_text="Normalized {} Distribution".format(normby),
        yaxis_title_text=y_axis_title,
    )

    return fig
