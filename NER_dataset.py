from datasets import load_dataset
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, dataset, tokenzier):
        super().__init__()
        self.dataset = dataset
        self.tokenzier = tokenzier

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        tokenized_sample = self.tokenize_and_align_labels(sample)
        return tokenized_sample
    
    def tokenize_and_align_labels(self, sample):
        tokenized_input = self.tokenzier(sample["tokens"], truncation=True, is_split_into_words=True)
        word_ids = tokenized_input.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(sample["ner_tags"][word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_input["labels"] = label_ids
        return tokenized_input
