import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Load BERT-based model for extractive summarization
model_name = "facebook/bart-large-cnn"  # Can be replaced with 'google/pegasus-xsum'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def split_sentences(text):
    """Splitting text into sentences using Hugging Face tokenizer instead of nltk."""
    return tokenizer.tokenize(text, add_special_tokens=False)

def extractive_summarization(text, num_sentences=3):
    # Tokenize into sentences without using nltk
    sentences = text.split(". ")  # A simple way to split without external libraries

    # Tokenize sentences and predict importance scores
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).cpu().numpy()

    # Get top N sentences with highest scores
    top_sentence_indices = np.argsort(scores)[-num_sentences:]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]

    return " ".join(top_sentences)

# Example usage
input_text = """Text summarization is usually implemented by natural language processing methods, 
designed to locate the most informative sentences in a given document. On the other hand, 
visual content can be summarized using computer vision algorithms. Image summarization is 
the subject of ongoing research; existing approaches typically attempt to display the most 
representative images from a given image collection, or generate a video that only includes 
the most important content from the entire collection. Video summarization algorithms 
identify and extract from the original video content the most important frames (key-frames), 
and/or the most important video segments (key-shots), normally in a temporally ordered 
fashion. Video summaries simply retain a carefully selected subset of the original video 
frames and, therefore, are not identical to the output of video synopsis algorithms, 
where new video frames are being synthesized based on the original video content."""

summary = extractive_summarization(input_text)
print("\nExtractive Summary:", summary)
