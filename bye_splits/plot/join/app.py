from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop

import pandas as pd
import numpy as np

from flask import Flask, render_template
app = Flask(__name__)

def bkapp(doc):
    df = pd.DataFrame({'x': [0,1,2,3,4,6],
                       'y': [0,1,2,3,4,5]})

    source = ColumnDataSource(df)

    plot = figure(title='Title')
    plot.line('x', 'y', source=source)

    doc.add_root(column(plot))
    #doc.theme = Theme(filename="theme.yaml")

@app.route('/bokeh', methods=['GET'])
def bkapp_page():
    if ssh_tunnel:
        ssh_path = 'http://localhost:{}/bkapp'
    else:
        ssh_path = 'http://llruicms01.in2p3.fr:{}/bkapp'
    script = server_document(ssh_path.format(bokeh_port))
    return render_template('embed.html', relative_urls=False,
                           script=script,
                           framework='Flask')

def bk_worker():
    if ssh_tunnel:
        web_socket = 'localhost:{}'
    else:
        web_socket = 'llruicms01.in2p3.fr:{}'
    server = Server({'/bkapp': bkapp}, io_loop=IOLoop(),
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
