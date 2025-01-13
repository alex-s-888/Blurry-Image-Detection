FROM  python:3.11-slim

RUN pip install pillow numpy tensorflow

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

ENTRYPOINT ["python", "inference.py"]
CMD ["input_dir", "output_file"]