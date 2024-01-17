# Import libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
import faiss
import numpy as np

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load vector database
vector_database = np.random.random((1000, 768)).astype('float32')  # Example: 1000 vectors with 768 dimensions

# Function to retrieve from the vector database
def retrieve_from_database(query_vector, top_k=5):
    index = faiss.IndexFlatL2(768)
    index.add(vector_database)
    _, retrieved_indices = index.search(np.array([query_vector]), top_k)
    return retrieved_indices.flatten()

# Function for RAG content generation
def generate_rag_content(query, top_k=5):
    # Tokenize the query
    input_ids = tokenizer.encode(query, return_tensors="pt")

    # Get hidden states from the language model
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state[0]

    # Convert hidden states to a numpy array
    query_vector = last_hidden_states.mean(dim=0).numpy()

    # Retrieve from the vector database
    retrieved_indices = retrieve_from_database(query_vector, top_k)

    # Generate content using GPT-2
    content_generator = pipeline('text-generation', model="gpt2")
    generated_content = []
    for index in retrieved_indices:
        context = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        generated_text = content_generator(context, max_length=50, num_return_sequences=1)[0]['generated_text']
        generated_content.append(generated_text)

    return generated_content

# Example usage
query = "Machine learning"
generated_content = generate_rag_content(query)

# Display the generated content
for i, content in enumerate(generated_content):
    print("Result {}: {}".format(i+1, content))

