from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_report(status, confidence):
    prompt = f"""
    Generate a short inspection report for a product image.
    Defect Status: {status}
    Confidence: {confidence:.2f}
    """

    response = generator(prompt, max_length=80, num_return_sequences=1)
    return response[0]["generated_text"]
