import os
import sys
import glob
import numpy as np
import tensorflow as tf
from PIL import Image


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


model = TFLiteModel("mymodel.tflite")
mylabels = ["Blurred", "Not Blurred"]


# Input data
input_dir = sys.argv[1]
# Placeholder for prediction
output_file_csv = sys.argv[2]

with open(output_file_csv, 'w') as my_output:

    for img_path in glob.glob(input_dir + '/*.jpg'):
        pil_image = Image.open(img_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert(mode='RGB')
        resized_image = pil_image.resize((256, 256))

        arr_image = np.asarray(resized_image)
        img = np.float32(arr_image / 255)

        label = mylabels[model.predict([img])[0].argmax()]
        my_output.write(os.path.basename(img_path) + ',' + label + '\n')
