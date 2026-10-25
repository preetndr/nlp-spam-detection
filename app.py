import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load the saved model and tokenizer
model = tf.keras.models.load_model("spam_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Same basic text cleaning function used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Sample messages taken directly from SPAM text message 20170820 - Data.csv
sample_messages = {
    "Type my own message...": "",
    "Ham Sample 1 (Not SPAM)": "Nah I don't think he goes to usf, he lives around here though",
    "Ham Sample 2 (Not SPAM)": "Is that seriously how you spell his name?",
    "Ham Sample 3 (Not SPAM)": "I'm back & we're packing the car now, I'll let you know if there's room",
    "Spam Sample 1 (SPAM)": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "Spam Sample 2 (SPAM)": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
    "Spam Sample 3 (SPAM)": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
}

# Simple Streamlit UI
st.title("Spam Message Detection App")
st.write("Pick a sample message from the dataset or type your own below to check whether it is **SPAM** or **Not SPAM (Ham)**.")

# Dropdown to pick a sample message
selected_option = st.selectbox("Choose a sample message to test:", list(sample_messages.keys()))

# Text area auto-fills with the selected sample (or stays blank for custom input)
user_input = st.text_area(
    "Message:",
    value=sample_messages[selected_option],
    placeholder="Type or paste a message here..."
)

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter or select a message first.")
    else:
        # Clean, tokenize, and pad the input text
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='post')

        # Predict using the trained model
        prediction = model.predict(padded)[0][0]

        if prediction >= 0.5:
            st.error(f"Result: **SPAM Message** (Confidence Score: {prediction:.2f})")
        else:
            st.success(f"Result: **Not SPAM (Ham)** (Confidence Score: {1 - prediction:.2f})")