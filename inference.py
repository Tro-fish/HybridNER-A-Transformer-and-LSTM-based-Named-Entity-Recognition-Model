import torch
from transformers import AutoTokenizer
from NER_models import ModelForNER
from utils import CONFIG, DotDict

def load_model(model_path, config, device):
    model = ModelForNER(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(text, model, tokenizer, label_list, device):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    ner_tags = [label_list[pred] for pred in predictions]

    result = list(zip(tokenized_text, ner_tags))
    return result

def main():
    config = DotDict(CONFIG)
    config.label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]  # Update with your label list if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    model = load_model("model/best_ner_model.pth", config, device)

    text = input("Enter the text: ")
    ner_results = predict(text, model, tokenizer, config.label_list, device)

    for token, label in ner_results:
        if label != 'O':  # Only print tokens with labels
            print(f"{token}: {label}")

if __name__ == "__main__":
    main()