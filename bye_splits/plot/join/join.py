from threading import Thread

from flask import Flask, render_template
from tornado.ioloop import IOLoop

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme

import pandas as pd

app = Flask(__name__)

def bkapp(doc):
    df = pd.DataFrame({'x': [0,1,2,3,4,5],
                       'y': [0,1,2,3,4,5]})
    source = ColumnDataSource(data=df)

    plot = figure(x_axis_type='x axis', y_axis_label='y axis', title='Title')
    plot.line('x', 'y', source=source)

    # def callback(attr, old, new):
    #     if new == 0:
    #         data = df
    #     else:
    #         data = df.rolling(f"{new}D").mean()
    #     source.data = ColumnDataSource.from_df(data)

    # slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    # slider.on_change('value', callback)

    #doc.add_root(column(slider, plot))
    doc.add_root(column(plot))

    #doc.theme = Theme(filename="theme.yaml")

@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('llruicms01.in2p3.fr:5006/bkapp')
    return render_template('embed.html', script=script, template='Flask')

def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': bkapp}, io_loop=IOLoop(),
                    allow_websocket_origin=['llruicms01.in2p3.fr:8010'])
    server.start()
    server.io_loop.start()

Thread(target=bk_worker).start()

if __name__ == '__main__':
    # print('Opening single process Flask app with embedded Bokeh application')
    # print()
    # print('Multiple connections may block the Bokeh app in this configuration!')
    # print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    app.run(host='llruicms01.in2p3.fr', port=8010, debug=False)


# import dash
# app = dash.Dash(__name__)
# server = app.server

# pass flask instance to dash
# import flask
# server = flask.Flask(__name__)
# app = dash.Dash(__name__, server=server)
# https://discourse.bokeh.org/t/embed-bokeh-server-app-in-a-flask-app/7556/4
