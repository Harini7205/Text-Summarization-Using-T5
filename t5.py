import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text, min_length=50, max_length=150, num_beams=5, top_k=50, top_p=0.95, temperature=0.7):
    # Ensure input is formatted correctly
    input_text = "summarize: " + text.strip().replace("\n", " ")

    # Tokenize input with truncation
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)

    # Generate summary with beam search & controlled randomness
    summary_ids = model.generate(
        inputs,
        min_length=min_length,
        max_length=max_length,
        num_beams=num_beams,  # Beam search for better quality
        no_repeat_ngram_size=3,  # Avoid word repetition
        top_k=top_k,  # Restrict word selection
        top_p=top_p,  # Nucleus sampling
        temperature=temperature  # Diversity in output
    )
    
    # Decode output
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

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

summary = summarize_text(input_text)
print("\nGenerated Summary:", summary)
