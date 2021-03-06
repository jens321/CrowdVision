import requests
import json
import cv2

def send_img(filename):
    addr = 'http://crowdvision.us-east-1.elasticbeanstalk.com'
    test_url = addr + '/api/test'

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = cv2.imread(filename)

    _, img_encoded = cv2.imencode('.jpg', img)

    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

    print(json.loads(response.text))

if __name__ == "__main__":
    send_img('study_room.jpg')