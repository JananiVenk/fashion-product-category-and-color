FROM python:3.7

RUN mkdir /code

WORKDIR /code

COPY ./app/category.pth /code/app/category.pth
RUN pip install scikit-learn==0.22.2
RUN pip install fast-colorthief
RUN pip install fastapi uvicorn torch opencv-python pillow pandas torchvision jinja2 python-multipart 

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip freeze > requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

ENV PYTHONPATH "${PYTHONPATH}:/code/app"

WORKDIR /code/app

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8012"]
