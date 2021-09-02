from flask import Flask, render_template, jsonify
from flask_classy import FlaskView
from flask_socketio import SocketIO, send, emit, Namespace
import threading
import time

# we'll make a list to hold some quotes for our app
quotes = [
    "A noble spirit embiggens the smallest man! ~ Jebediah Springfield",
    "If there is a way to do it better... find it. ~ Thomas Edison",
    "No one knows what he can do till he tries. ~ Publilius Syrus"
]

app = Flask(__name__)

class server(FlaskView, threading.Thread):
    
    def __init__(self):
        # self.app = Flask("FlaskTemplate")
        threading.Thread.__init__(self)
        self.keep_alive = True
        self.socketio = SocketIO(app)
        self.estadoAdelante = "on"
        self.atras = "on"
        self.izquierda = "on"
        self.derecha = "on"    
        # self.socketio.run(self.app)
        
    def index(self):
        # return "<br>".join(quotes)
        return render_template('index.html')
    
    def run(self):

        sleep_time = 1 #secs
        
        while self.keep_alive:

            app.run()
            time.sleep(sleep_time)

server.register(app)

if __name__ == '__main__':
    servidor = server()
    servidor.start()
    try:
        time.sleep(4)
        
    finally:
        servidor.keep_alive = False
        servidor.join() #free thread

    # app.run()