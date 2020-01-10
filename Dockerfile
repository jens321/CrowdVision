FROM python:3.6

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# install dependencies
RUN apt-get update && apt-get install -y \
  libsm6 \
  libxrender1 \
  libfontconfig1 \
  libice6

RUN pip3 install --upgrade Cython
RUN pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python3", "./main.py"]