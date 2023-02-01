from dash import html
import dash

dash.register_page(__name__, name='Home')

layout = html.H1("Please select an app that you would like to use.")
