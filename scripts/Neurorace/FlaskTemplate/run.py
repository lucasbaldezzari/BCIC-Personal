from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, send, emit, Namespace
from flask_classy import FlaskView

import threading

app = Flask(__name__)

class servidor():

    # ESTO ES EL SERVIDOR CENTRAL.
    
    def __init__(self):
        # self.app = Flask("FlaskTemplate")
        self.socketio = SocketIO(app)
        self.estadoAdelante = "on"
        self.atras = "on"
        self.izquierda = "on"
        self.derecha = "on"    
        self.socketio.run(self.app)
        
    @app.route('/')
    def start(self):
        # self.app.route('/')
        render_template('index.html')
        # self.socketio.run(self.app)
    
    def estimAdelante(self):
        self.app.route('/adelante')
        return jsonify(message = self.estadoAdelante)
       
    def behind(self):
        self.app.route('/atras')
        return jsonify(message = self.atras)
    
    def left(self):
        self.app.route('/left')
        return jsonify(message = self.izquierda)
    
    def rigth(self):
        self.app.route('/rigth')
        return jsonify(message = self.derecha)

# servidor.register(app)

if __name__ == '__main__':
    app.run()
    
     