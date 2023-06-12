import torch
from transformers import BertTokenizer
from Bert_test_1 import Model

# Load the downstream task model and move it to the GPU (if available)
model = Model()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to preprocess the input
def preprocess_input(input_text):
    input_text = input_text.strip()  # remove leading/trailing whitespace
    input_encoded = tokenizer.encode_plus(
        input_text,
        padding='max_length',
        truncation=True,
        max_length=500,
        return_tensors='pt'
    )
    input_ids = input_encoded['input_ids']
    attention_mask = input_encoded['attention_mask']
    token_type_ids = input_encoded['token_type_ids']
    return input_ids, attention_mask, token_type_ids

# Define a function to make predictions
def predict_slither(input_text):
    input_ids, attention_mask, token_type_ids = preprocess_input(input_text)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        proba = torch.sigmoid(logits)
        pred = (proba > 0.5).int()
    return pred.tolist()[0]

# Example usage:
source_code = "function add(uint256 a, uint256 b) public pure returns (uint256 c) { c = a + b; }"
pred = predict_slither(source_code)
print(pred)
