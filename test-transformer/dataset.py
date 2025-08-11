import random
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_indices = [
            self.src_vocab.get(word, self.src_vocab["<unk>"]) for word in src.split()
        ]
        tgt_indices = [
            self.tgt_vocab.get(word, self.tgt_vocab["<unk>"]) for word in tgt.split()
        ]

        src_indices = (
            [self.src_vocab["<sos>"]] + src_indices + [self.src_vocab["<eos>"]]
        )
        tgt_indices = (
            [self.tgt_vocab["<sos>"]] + tgt_indices + [self.tgt_vocab["<eos>"]]
        )

        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.split())

    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch


def get_sample_data():
    data_pairs = [
        ("hello", "ciao"),
        ("good morning", "buongiorno"),
        ("good evening", "buonasera"),
        ("good night", "buonanotte"),
        ("thank you", "grazie"),
        ("please", "per favore"),
        ("yes", "sì"),
        ("no", "no"),
        ("how are you", "come stai"),
        ("I am fine", "sto bene"),
        ("what is your name", "come ti chiami"),
        ("my name is John", "mi chiamo John"),
        ("nice to meet you", "piacere di conoscerti"),
        ("goodbye", "arrivederci"),
        ("see you later", "a dopo"),
        ("excuse me", "scusi"),
        ("I am sorry", "mi dispiace"),
        ("I love you", "ti amo"),
        ("where is the bathroom", "dov'è il bagno"),
        ("how much does it cost", "quanto costa"),
        ("I don't understand", "non capisco"),
        ("do you speak English", "parli inglese"),
        ("I speak Italian", "parlo italiano"),
        ("water please", "acqua per favore"),
        ("coffee please", "caffè per favore"),
        ("one two three", "uno due tre"),
        ("the book is red", "il libro è rosso"),
        ("the house is big", "la casa è grande"),
        ("I like pizza", "mi piace la pizza"),
        ("where are you from", "di dove sei"),
        ("I am from America", "vengo dall'America"),
        ("what time is it", "che ore sono"),
        ("it is late", "è tardi"),
        ("it is early", "è presto"),
        ("the weather is nice", "il tempo è bello"),
        ("it is raining", "sta piovendo"),
        ("I am hungry", "ho fame"),
        ("I am thirsty", "ho sete"),
        ("the food is delicious", "il cibo è delizioso"),
        ("can you help me", "puoi aiutarmi"),
        ("I need help", "ho bisogno di aiuto"),
        ("left right straight", "sinistra destra dritto"),
        ("open the door", "apri la porta"),
        ("close the window", "chiudi la finestra"),
        ("come here", "vieni qui"),
        ("go away", "vai via"),
        ("wait a moment", "aspetta un momento"),
        ("I understand now", "ora capisco"),
        ("speak slowly please", "parla lentamente per favore"),
        ("write it down", "scrivilo"),
    ]

    random.shuffle(data_pairs)

    src_sentences = [pair[0] for pair in data_pairs]
    tgt_sentences = [pair[1] for pair in data_pairs]

    src_vocab = build_vocab(src_sentences)
    tgt_vocab = build_vocab(tgt_sentences)

    train_size = int(0.8 * len(data_pairs))
    train_pairs = data_pairs[:train_size]
    val_pairs = data_pairs[train_size:]

    return train_pairs, val_pairs, src_vocab, tgt_vocab


def get_dataloaders(batch_size=2):
    train_pairs, val_pairs, src_vocab, tgt_vocab = get_sample_data()

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, src_vocab, tgt_vocab
