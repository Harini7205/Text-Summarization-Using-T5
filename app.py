import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0)) 

# Load model and tokenizer
MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit UI
st.title("📝 Text Summarization with T5")
st.subheader("Enter your text below and get a concise summary!")

# Text input
user_input = st.text_area("Enter text to summarize:", height=200)

# Hyperparameters selection
st.sidebar.header("⚙️ Adjust Hyperparameters")
min_length = st.sidebar.slider("Min Summary Length", 20, 100, 50)
max_length = st.sidebar.slider("Max Summary Length", 50, 300, 150)
num_beams = st.sidebar.slider("Beam Search (Higher = Better Quality)", 1, 10, 5)
top_k = st.sidebar.slider("Top-K Sampling", 0, 100, 50)
top_p = st.sidebar.slider("Top-P Nucleus Sampling", 0.0, 1.0, 0.95)
temperature = st.sidebar.slider("Temperature (Higher = More Randomness)", 0.1, 1.5, 0.7)
do_sample= st.sidebar.checkbox("Enable Sampling", value=True)

# Summarization button
if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess input
        input_text = "summarize: " + user_input.strip().replace("\n", " ")
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        # Generate summary
        summary_ids = model.generate(
            inputs,
            do_sample=do_sample,
            min_length=min_length,
            max_length=max_length,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=3
        )

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display summary
        st.subheader("📌 Generated Summary:")
        st.write(summary)

