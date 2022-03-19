import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import wandb
import string
import ftfy


class SentenceDataset(Dataset):
    def __init__(self, seq_len, data_file):
        super().__init__()

        self.data = []
        self.vocab = {'<unk>': 0}
        self.seq_len = seq_len

        # Pull data from the given dataset file, and run the following
        # preprocessing rules:
        #
        # 1. Lowercase all text.
        # 2. Remove encoding error using ftfy.
        # 3. Remove punctuation and formatters e.g '\t' '\n'.
        # 4. Split on words and build vocab.
        self.load_and_preprocess(data_file)

    def load_and_preprocess(self, data_file):
        with open(data_file, 'r') as f:
            raw_data = f.read().lower()
            raw_data = ftfy.fix_text(raw_data)

            # Let's get rid of punctuation to simplify things a little.
            seq_to_remove = [w for w in string.punctuation]
            for w in seq_to_remove:
                raw_data = raw_data.replace(w, '')
            for w in ['\t', '\n']:
                raw_data = raw_data.replace(w, ' ')

            # Our dataset is now a long list of words.
            self.data = raw_data.split(' ')

            # Add the word to our vocab if it doesn't already exist.
            for word in self.data:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def words_to_idx(self, words):
        return [self.word_to_idx(w) for w in words]

    def word_to_idx(self, w):
        return self.vocab.get(w, self.vocab['<unk>'])

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.words_to_idx(self.data[idx:idx + self.seq_len])
        y = self.words_to_idx(self.data[idx+1:idx+1 + self.seq_len])

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class Trainer:
    """
    Trainer is a generic model trainer that, given a model and a dataset,
    will train and test the model until a sufficiently low loss is reached.
    """
    def __init__(self, model, config, train_dataset, test_dataset):
        self.model = model
        self.config = config
        self.train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

    def train(self):
        self.model.train()

        for epoch in range(self.config.epochs):
            for i, (batch_x, batch_y) in enumerate(self.train_data):
                with torch.set_grad_enabled(True):
                    y_hat, loss = self.model(batch_x, batch_y)
                    if i % self.config.print_loss_every_iter == 0:
                        print("Loss: ", loss)

                if self.config.logging:
                    wandb.log({"loss": loss})

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch > 0 and epoch % self.config.test_every_n_epochs == 0:
                self.test()

    def test(self):
        self.model.eval()

        if not self.test_data or len(self.test_data) == 0:
            return

        av_loss = 0
        for i, (batch_x, batch_y) in enumerate(self.test_data):
            _, loss = self.model(batch_x, batch_y)
            av_loss += loss

        av_loss /= len(self.test_data)
        print(f"Average test loss: {av_loss}")

        if self.config.logging:
            wandb.log({"test_loss": av_loss})
