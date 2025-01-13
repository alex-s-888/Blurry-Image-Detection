import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


model = TFLiteModel("../deployment_docker/mymodel.tflite")
mylabels = ["Blurred", "Not Blurred"]


def predict_blur(input_image):

    print(datetime.datetime.now())

    pilImage = Image.fromarray(input_image)
    if pilImage.mode != 'RGB':
        pilImage = pilImage.convert(mode='RGB')
    resized_image = pilImage.resize((256, 256))

    arr_image = np.asarray(resized_image)
    img = np.float32(arr_image/255)

    label = mylabels[ model.predict([img])[0].argmax() ]
    return "Prediction: " + label

demo = gr.Interface(
    fn = predict_blur,
    inputs = ["image"],
    outputs = ["text"],
)

demo.launch()