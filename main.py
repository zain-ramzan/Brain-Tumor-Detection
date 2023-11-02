import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf



model = tf.keras.models.load_model('zain_model.h5')

st.title('Brain Tumor Detection')
# Display the main message
st.write("You can see the tumor in an MRI brain image if any. Upload an image and see!")
# Create a link to an external website

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB").resize((250, 250))
    image_array = np.array(image)
    prediction = model.predict(np.array([image_array]))
    predicted_digit = np.argmax(prediction)

    tumor_types = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
    predicted_class = tumor_types[predicted_digit]

    # st.subheader(f'Tumor Type: {predicted_class}')
    st.markdown(
    """
    <div style="text-align:center;">
    <h4 style='background-color: #f0f2f6; color: #31333f; padding: 5px; border-radius: 5px;'> 
    """
    f'{predicted_class}'
    
    """
    </h4>
    </div>
    """,
    unsafe_allow_html=True,
    )
    
    st.image(image, use_column_width=False, width=200)
with st.container():
    st.markdown(
        """
        <div style='background-color: #f0f2f6; color: #31333f; padding: 10px; border-radius: 5px;'> 
        Design with &#10084; by <a style="color:#31333f;" href="https://zain-ramzan.github.io">Zain Ramzan</a>
        </div>
        """,
        unsafe_allow_html=True,

        )
