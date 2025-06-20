import tensorflow as tf

# Load your .keras model
keras_model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the .tflite model to a file
with open('trained_plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
