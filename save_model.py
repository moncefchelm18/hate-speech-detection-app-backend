import tensorflow as tf

model = tf.keras.models.load_model('hate_speech_model')  # Load the model
print("Model loaded successfully.")

print("Saving the model...")
model.save('hate_speech_model.h5')
print("Model saved successfully.")