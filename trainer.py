import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import random
import wandb


class MathDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.vocab = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "+": 10,
            "=": 11,
            "#": 12,  # Mask char.
        }

        self.items = [self.gen_item() for _ in range(self.__len__())]

    def vocab_len(self):
        return len(self.vocab)

    def problem_to_idx(self, problem):
        idx_list = []
        for char in problem:
            if char == " ":
                continue
            idx_list.append(self.char_to_idx(char))
        return idx_list

    def char_to_idx(self, char):
        return self.vocab[char]

    def idx_to_char(self, idx):
        for char, i in self.vocab:
            if idx == i:
                return char

        return None

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.items[idx]

    def gen_item(self):
        """
            Firstly we generate some equation such as "5 + 5 = 10".
            Then we build n (max=3) examples from the equation. e.g

            x: [5 + 5] y: [=]
            x: [5 + 5 =] y: [1]
            x: [5 + 5 = 1] y: [0]

            We also pad x with # in all positions that it shouldn't
            attend, so it doesn't look ahead to cheat. This is known as masking.
        """
        digit_choices = [i for i in range(1, 10)]

        n1 = random.choice(digit_choices)
        n2 = random.choice(digit_choices)
        n_sum = str(n1 + n2)

        problem = self.problem_to_idx("{} + {} = {}".format(n1, n2, n_sum.rjust(2, '0')))

        x = []
        y = []
        for (i, idx) in enumerate(problem[::-1]):
            x_temp = problem[:len(problem) - i - 1]
            for _ in range(i):
                x_temp.append(self.char_to_idx("#"))  # Add a mask.

            x.append(x_temp)
            y.append([problem[len(problem)-i-1]])

            if idx == self.char_to_idx("="):
                break

        lucky_dip = random.randint(0, len(x)-1)
        x_tensor = torch.tensor(x[lucky_dip], dtype=torch.long)
        y_tensor = torch.tensor(y[lucky_dip], dtype=torch.long)
        return x_tensor, y_tensor


class TrainerConfig:
    epochs = 120
    batch_size = 32
    learning_rate = 3e-4

    print_loss_every = 10
    test_every_n_epochs = 10

    # Log to wandb.
    logging = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    """
    Trainer is a generic model trainer that, given a model and a dataset,
    will train and test the model until a sufficiently low loss is reached.
    """
    def __init__(self, model, train_dataset, test_dataset, config):
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
                    if i % self.config.print_loss_every == 0:
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
