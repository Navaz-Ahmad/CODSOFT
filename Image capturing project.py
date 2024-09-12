import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained ResNet50 model for image feature extraction
def load_image_model():
    # We will remove the final classification layer to get features
    base_model = ResNet50(weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load and preprocess the image
def preprocess_image(img_path):
    # Load the image with a target size of (224, 224) as required by ResNet
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to an array
    img_data = image.img_to_array(img)
    # Expand dimensions to match ResNet input shape
    img_data = np.expand_dims(img_data, axis=0)
    # Preprocess the image (mean subtraction, scaling, etc.)
    return preprocess_input(img_data)

# Create a simple LSTM-based model for caption generation
def create_caption_model(vocab_size, max_caption_length):
    # Define image features input (from ResNet)
    image_input = Input(shape=(2048,))
    # Add a dropout layer to prevent overfitting
    img_features = Dropout(0.5)(image_input)
    img_features = Dense(256, activation='relu')(img_features)

    # Define caption input (sequences)
    caption_input = Input(shape=(max_caption_length,))
    # Add an embedding layer for the caption words
    caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    # Add LSTM layers to process the captions
    caption_lstm = LSTM(256)(caption_embedding)

    # Combine the image features and caption LSTM output
    combined = tf.keras.layers.add([img_features, caption_lstm])
    combined = Dense(256, activation='relu')(combined)

    # Final dense layer to predict the next word in the caption
    output = Dense(vocab_size, activation='softmax')(combined)

    # Create the model
    return Model(inputs=[image_input, caption_input], outputs=output)

# Generate a caption for the given image using greedy search
def generate_caption(model, image_features, tokenizer, max_caption_length):
    # Initialize the caption sequence with the start token
    caption = ['startseq']
    for _ in range(max_caption_length):
        # Convert the current caption to a sequence of integers
        seq = tokenizer.texts_to_sequences([caption])[0]
        # Pad the sequence to the maximum length
        seq = pad_sequences([seq], maxlen=max_caption_length)
        
        # Predict the next word
        y_pred = model.predict([image_features, seq], verbose=0)
        y_pred = np.argmax(y_pred)

        # Convert the predicted integer back to a word
        word = tokenizer.index_word.get(y_pred, None)
        if word is None:
            break
        
        # Append the word to the caption
        caption.append(word)
        
        # Stop if the end token is generated
        if word == 'endseq':
            break

    # Join the words to form the final caption, and return it
    return ' '.join(caption[1:-1])

# Main function to run the image captioning model
def image_captioning_pipeline(img_path, tokenizer, max_caption_length, vocab_size):
    # Load the image model (ResNet50)
    image_model = load_image_model()

    # Preprocess the input image and extract features
    img_data = preprocess_image(img_path)
    img_features = image_model.predict(img_data)

    # Load or create the caption model
    caption_model = create_caption_model(vocab_size, max_caption_length)

    # Use a tokenizer to convert words to sequences (this would be pre-trained)
    # For simplicity, we're assuming the tokenizer is available and trained on the caption dataset.

    # Generate a caption for the image
    caption = generate_caption(caption_model, img_features, tokenizer, max_caption_length)
    
    # Print the resulting caption
    print("Generated Caption: ", caption)

# Assume we have a trained tokenizer and a max_caption_length (example)
# In practice, you would load this from your training data
vocab_size = 5000  # Example vocabulary size
max_caption_length = 35  # Example max caption length
tokenizer = ...  # Assume this is trained

# Path to the input image
img_path = 'path_to_image.jpg'

# Run the image captioning pipeline
image_captioning_pipeline(img_path, tokenizer, max_caption_length, vocab_size)
