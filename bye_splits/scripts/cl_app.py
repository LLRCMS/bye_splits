from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SOLAR])
load_figure_template('SOLAR')

sidebar = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(page["name"], className='ms-2')
                    ],
                    href=page['relative_path'],
                    active='exact'
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            className='bg-light'
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div("Cluster Studies",
                        style={'fontSize': 50, 'textAlign': 'center'}))
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col(
        [
            sidebar
        ], xs=4, md=2, lg=2, xl=2, xxl=2
        ),

        dbc.Col(
        [
            dash.page_container
        ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10
        )
    ])
], fluid=True)

if __name__ == '__main__':
    import socket

    host, port = socket.gethostbyname(socket.gethostname()), 8052

    app.run_server(port=port, host=host,debug=True)
