import numpy as np
from scipy.integrate import odeint
import pandas as pd
import argparse

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop

from flask import Flask, render_template, render_template_string
app = Flask(__name__)

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from flask import Flask, render_template_string
import pandas as pd
import plotly.express as px

app = Flask(__name__)

def bokeh_app(doc):
    """
    Example available at https://docs.bokeh.org/en/3.2.2/docs/examples/basic/lines/lorenz.html
    """
    sigma = 10
    rho = 28
    beta = 8.0/3
    theta = 3 * np.pi / 4

    def lorenz(xyz, t):
        x, y, z = xyz
        x_dot = sigma * (y - x)
        y_dot = x * rho - x * z - y
        z_dot = x * y - beta* z
        return [x_dot, y_dot, z_dot]

    initial = (-10, -7, 35)
    t = np.arange(0, 100, 0.006)

    solution = odeint(lorenz, initial, t)

    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]
    xprime = np.cos(theta) * x - np.sin(theta) * y

    colors = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"]

    plot = figure(title="Lorenz attractor example", background_fill_color="#fafafa")
    
    plot.multi_line(np.array_split(xprime, 7), np.array_split(z, 7),
                    line_color=colors, line_alpha=0.8, line_width=1.5)

    doc.add_root(column(plot))

def plotly_app():
    df = pd.DataFrame({'x': [1,2,5,6,7,8,3,4,7,2],
                       'y': [1,2,5,6,7,8,8,9,2,5],
                       'z': [1,2,5,5,2,1,8,3,5,2]})
    fig = px.scatter_3d(df,
                        x='z', y='x', z='y',
                        color_discrete_sequence=['black'],
                        symbol_sequence=['circle'],
                        hover_data=['x', 'y', 'z'],
                        )
    return fig.to_html(full_html=False)

@app.route('/viz', methods=['GET'])
def webapp_page():
    if FLAGS.ssh:
        ssh_path = 'http://localhost:{}/bokeh_app'
    else:
        ssh_path = 'http://llruicms01.in2p3.fr:{}/bokeh_app'
    script_bokeh = server_document(ssh_path.format(FLAGS.bokeh_port))

    script_plotly = plotly_app() #not really an app, rather an html block

    return render_template('embed.html', relative_urls=False,
                           script_bokeh=script_bokeh,
                           script_plotly=script_plotly,
                           framework='Flask')

def bk_worker():
    web_socket = 'localhost:{}' if FLAGS.ssh else 'llruicms01.in2p3.fr:{}'
    server = Server({'/bokeh_app': bokeh_app}, io_loop=IOLoop(),
                    port=FLAGS.bokeh_port,
                    allow_websocket_origin=[web_socket.format(FLAGS.flask_port)])
    server.start()
    server.io_loop.start()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run example: `python bye_splits/plot/join/app.py`')
    p.add_argument('--bokeh_port', required=False, default=8008, type=int,
                   help='Specify port for bokeh server.')
    p.add_argument('--flask_port', required=False, default=8010, type=int,
                   help='Specify port for flask server.')
    p.add_argument('--ssh', action='store_true',
                   help='Whether to serve ports with a SSH connection in mind.')
    FLAGS = p.parse_args()
    
    from threading import Thread
    Thread(target=bk_worker).start()

    app.run(port=FLAGS.flask_port, host='llruicms01.in2p3.fr', debug=False)
