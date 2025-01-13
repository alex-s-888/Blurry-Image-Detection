
import os
import zipfile
from tensorflow import keras

# Download and unzip dataset
dataset_url = "https://github.com/alex-s-888/Blurry-Image-Detection/raw/refs/heads/main/dataset/blur-noblur.zip"
zip_file_path = "../dataset/blur-noblur.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
# Extract to same name without the .zip extension
extract_dir = os.path.splitext(zip_file_path)[0]
# Extract the files
zip_ref.extractall(extract_dir)
zip_ref.close()

# Print the path to the extracted files
print(f"Dataset extracted to: {extract_dir}")


BATCH_SIZE = 24

img_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

train_ds = img_gen.flow_from_directory(
    extract_dir + '/train',
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical'
)

print(train_ds.class_indices)

val_ds = img_gen.flow_from_directory(
    extract_dir + '/validation',
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)


from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPooling2D,Input,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

def prepare_model(learning_rate=0.001, dropout = 0.2):
  model = Sequential()

  model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)))
  model.add(MaxPooling2D())
  model.add(Conv2D(32, 3, padding='same', activation='relu'))
  model.add(MaxPooling2D())
  model.add(Conv2D(64, 3, padding='same', activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  if dropout > 0:
    model.add(Dropout(dropout))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
  )

  return model


EPOCHS = 12


# Best learning_rate is 0.001, build the model
model = prepare_model(learning_rate=0.001)

history = model.fit(
  train_ds,
  epochs=EPOCHS,
  validation_data=val_ds
)

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('../deployment_docker/mymodel.tflite', 'wb') as f_out:
    f_out.write(tflite_model)