from flask import Flask, request, Response
import jsonpickle
import sys 
import resource
import numpy as np
import cv2
import os 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
print("Instantiating model ...")
predictor = DefaultPredictor(cfg)
print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print("type of model", type(predictor))

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/api/test', methods=['POST'])
def test():
    r = request

    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    outputs = predictor(img)
    print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # info_dict = outputs['instances'].get_fields()
    # info_dict['pred_boxes'] = info_dict['pred_boxes'][[2, 3, 5, 8]]
    # info_dict['scores'] = info_dict['scores'][[2, 3, 5, 8]]
    # info_dict['pred_classes'] = info_dict['pred_classes'][[2, 3, 5, 8]]
    # info_dict['pred_masks'] = info_dict['pred_masks'][[2, 3, 5, 8]]

    # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite('study_room_analysed_persons.jpg', v.get_image()[:, :, ::-1])

    # build response dict to send back to client
    response = {
        'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
        'count': len([el for el in outputs["instances"].pred_classes if el == 0])
    }
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))