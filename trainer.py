import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import wandb
import json
from typing import List
import datetime

class EventDataset(Dataset):
    def __init__(self, seq_len, data_file, device):
        super().__init__()

        self.data = []
        self.vocab = {'<unk>': 0}
        self.seq_len = seq_len
        self.device = device

        self.load_and_preprocess(data_file)

    def load_and_preprocess(self, data_file):
        with open(data_file, 'r', encoding='utf-8-sig') as f:
            self.data = json.load(f)

            for event in self.data:
                event = str(event)
                if not self.vocab.get(event):
                    self.vocab[event] = len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def idx_to_event(self, event):
        for (k, v) in enumerate(self.vocab):
            if k == event:
                return v
        return None

    def events_to_idx(self, events: List[str]):
        return [self.event_to_idx(e) for e in events]

    def event_to_idx(self, e):
        return self.vocab[str(e)]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.events_to_idx(self.data[idx:idx + self.seq_len])
        y = self.events_to_idx(self.data[idx+1:idx+1 + self.seq_len])

        return (
            torch.tensor(x, dtype=torch.long).to(self.device),
            torch.tensor(y, dtype=torch.long).to(self.device),
        )


class Trainer:
    """
    Trainer is a generic model trainer that, given a model and a dataset,
    will train and test the model until a sufficiently low loss is reached.
    """
    def __init__(self, model, config, train_dataset, test_dataset=None):
        self.model = model
        self.config = config
        self.train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        self.test_data = None
        if test_dataset:
            self.test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

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

            if self.test_data and epoch > 0 and epoch % self.config.test_every_n_epochs == 0:
                self.test()

            # Save a checkpoint at each epoch.
            if epoch % self.config.save_chkpt_every_n_epochs == 0:
                fn = datetime.datetime.now().isoformat()
                path = '/Users/farleyschaefer/Documents/projects/newco/engine/models'
                torch.save(self.model.state_dict(), f"{path}/{fn}.pt")

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
