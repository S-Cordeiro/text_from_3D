"""GPT-2 BPE tokenizer: use tiktoken's existing vocabulary, without retraining it."""
import tiktoken


class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding('gpt2')
        self.vocab_size = self.enc.n_vocab
        self.eot_token = self.enc.eot_token

    def encode(self, text, add_eot=False):
        # Special-token spellings in user text are treated as ordinary text.
        tokens = self.enc.encode_ordinary(text)
        return tokens + [self.eot_token] if add_eot else tokens

    def decode(self, tokens):
        return self.enc.decode(list(tokens))
