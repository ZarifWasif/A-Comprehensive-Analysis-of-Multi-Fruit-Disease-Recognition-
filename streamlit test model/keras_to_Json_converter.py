import tensorflow as tf

# Load the .keras model
model = tf.keras.models.load_model('/trained_plant_disease_model.keras')

# Convert model architecture to JSON
model_json = model.to_json()

# Save the JSON to a file
with open('trained_plant_disease_model.json', 'wb') as json_file:
    json_file.write(model_json)

print("Model architecture saved to model.json")

