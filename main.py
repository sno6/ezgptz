import torch

from model import GPT, Config
from trainer import SentenceDataset, Trainer

if __name__ == '__main__':
    config = Config(logging=False)

    # Set up wandb for model performance logging.
    if config.logging:
        import wandb
        wandb.init(project="ezgptz", entity="sno6")
        wandb.config = {
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        }

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    print(f'Running model on device: {device}')

    train_dataset = SentenceDataset(seq_len=config.seq_len, data_file='./data/training.txt', device=device)

    model = GPT(config, vocab_len=len(train_dataset.get_vocab())).to(device)
    trainer = Trainer(model, config, train_dataset, None)
    trainer.train()
