import torch
# Load the question-answering pipeline
from transformers import pipeline
question_answerer = pipeline('question-answering')

device=torch.device("mps")
print(f"device={device}")
# Define a context and question
context = "The quick brown fox jumps over the lazy dog"
question = "What animal jumps over the lazy dog?"

# Use the pipeline to answer the question
answer = question_answerer(question=question, context=context, device=device)

# Print the answer
# print context 
print(f'context={context}')
print(f'question={question}')
print(f'answer={answer}')
