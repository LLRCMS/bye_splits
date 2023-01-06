from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop

import pandas as pd
import numpy as np

from flask import Flask, render_template, render_template_string
app = Flask(__name__)

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from flask import Flask, render_template_string
import pandas as pd
import plotly.express as px

app = Flask(__name__)

def bokeh_app(doc):
    df = pd.DataFrame({'x': [0,1,2,3,4,6],
                       'y': [0,1,2,3,4,5]})

    source = ColumnDataSource(df)

    plot = figure(title='Title')
    plot.line('x', 'y', source=source)

    doc.add_root(column(plot))
    #doc.theme = Theme(filename="theme.yaml")

def plotly_app():
    df = pd.DataFrame({'x': [1,2,5,6,7,8],
                       'y': [1,2,5,6,7,8],
                       'z': [1,2,5,6,7,8]})
    fig = px.scatter_3d(df,
                        x='z', y='x', z='y',
                        color_discrete_sequence=['black'],
                        symbol_sequence=['circle'],
                        hover_data=['x', 'y', 'z'],
                        )
    return fig.to_html(full_html=False)

@app.route('/viz', methods=['GET'])
def webapp_page():
    if ssh_tunnel:
        ssh_path = 'http://localhost:{}/bokeh_app'
    else:
        ssh_path = 'http://llruicms01.in2p3.fr:{}/bokeh_app'
    script_bokeh = server_document(ssh_path.format(bokeh_port))

    script_plotly = plotly_app() #not really an app, rather an html block

    return render_template('embed.html', relative_urls=False,
                           script_bokeh=script_bokeh,
                           script_plotly=script_plotly,
                           framework='Flask')

def bk_worker():
    if ssh_tunnel:
        web_socket = 'localhost:{}'
    else:
        web_socket = 'llruicms01.in2p3.fr:{}'
    server = Server({'/bokeh_app': bokeh_app}, io_loop=IOLoop(),
                    port=bokeh_port,
                    allow_websocket_origin=[web_socket.format(flask_port)])
    server.start()
    server.io_loop.start()

if __name__ == '__main__':
    ssh_tunnel = True
    bokeh_port = 8008
    flask_port = 8010

    from threading import Thread
    Thread(target=bk_worker).start()

    app.run(port=flask_port, host='llruicms01.in2p3.fr', debug=False)
