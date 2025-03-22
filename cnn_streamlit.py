import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

model = load_model("cnn_model_v2.keras")

st.title("Digit Recognition using CNN")

canvas_result = st_canvas(
    fill_color="black",
    stroke_color="white",
    stroke_width=15,
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

submit_button = st.button("Submit Drawing")

if submit_button and canvas_result.image_data is not None:
    
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    image = image.convert('L')
    
    image_array = np.array(image)
    threshold_value = 128
    image_array[image_array >= threshold_value] = 255

    image = Image.fromarray(image_array)

    image = image.resize((28, 28))
    image_array = np.array(image)

    image_array = image_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Processed Image (Input to Model)", width=150, clamp=True, channels="L")

    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions)

    st.write(f"### Predicted Digit: {predicted_digit}")


