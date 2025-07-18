import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
model = tf.keras.models.load_model('sms_spam_cnn_model.h5')

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
def preprocess_input(text,maxlen=100):
    sequence=tokenizer.texts_to_sequences([text])
    padded=pad_sequences(sequence,maxlen=maxlen)
    return padded
st.title("Spam SMS Detector")
user_input=st.text_area("Enter Your SMS message: ")
if st.button("Check"):
    if user_input.strip()=="":
        st.warning("Please enter a valid SMS")
    else:
        processed=preprocess_input(user_input)
        prediction=model.predict(processed)[0][0]
        
        # results
        if prediction > 0.5:
            st.error("🚫 This message is likely *Spam*!")
        else:
            st.success("✅ This message is likely *Not Spam* (Ham).")
