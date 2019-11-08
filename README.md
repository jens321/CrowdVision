# CrowdVision
CrowdVision is a project that aims to count the number of people in an image taken by a Raspberry Pi. This is a general task that can be used for many purposes. In our case, we will use this system to provide UCI students with occupancy data through means of a web app. This way we hope students will have an easier time finding a study place that works for them. 

## Technology
Once our Pi takes a shot of a study room, we sent the image to a Python [Flask Server](https://flask.palletsprojects.com/en/1.1.x/) and use Facebook's [Detectron2](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/) to count the number of people in the image. The image is then deleted and the data is stored in our PostgreSQL database (TODO). 