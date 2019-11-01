from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/api/test', methods=['POST'])
def test():
    r = request

    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # build response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run()