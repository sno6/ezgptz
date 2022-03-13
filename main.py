from model import GPT, GPTConfig
from trainer import MathDataset, Trainer, TrainerConfig
import torch

if __name__ == '__main__':
    dataset = MathDataset()

    trainer_cfg = TrainerConfig()
    model_cfg = GPTConfig(vocab_len=dataset.vocab_len())
    model = GPT(model_cfg)

    trainer = Trainer(model, dataset, trainer_cfg)
    trainer.train()

    model.eval()

    for i in range(10):
        ex_x, ex_y = dataset.gen_item()
        ex_x = ex_x.unsqueeze(0)
        ex_x = ex_x.unsqueeze(0)
        ex_y = ex_y.unsqueeze(0)

        y_hat, y_loss = model(ex_x, ex_y)

        print("Given")
        print(ex_x)
        print("Expected")
        print(ex_y)
        print("Got")
        print(torch.argmax(y_hat))
        print("With loss")
        print(y_loss)

        print("\n")
