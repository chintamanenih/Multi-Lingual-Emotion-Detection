import streamlit as st
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch


# Load model and tokenizer
@st.cache_resource
def load_model():
    model = XLMRobertaForSequenceClassification.from_pretrained("./finetuned_model")
    tokenizer = XLMRobertaTokenizer.from_pretrained("./finetuned_model")
    return model, tokenizer

model, tokenizer = load_model()

# Define the label mapping
label_mapping = {0: "anger", 1: "fear", 2: "joy", 3: "sadness", 4: "surprise"}

# Streamlit app title and description
st.title("Emotion Detection App")
st.write("Enter a sentence, and the model will predict the emotion.")

# User input
user_input = st.text_input("Type a sentence here...")

# Predict emotion
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Ensure model and inputs are on the same device
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_emotion = label_mapping[predicted_label]

    # Display the result
    st.write(f"**Predicted Emotion:** {predicted_emotion}")
