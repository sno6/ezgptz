from model import GPT, Config
from trainer import SentenceDataset, Trainer

if __name__ == '__main__':
    config = Config()

    # Set up wandb for model performance logging.
    if config.logging:
        import wandb
        wandb.init(project="ezgptz", entity="sno6")
        wandb.config = {
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        }

    train_dataset = SentenceDataset(seq_len=config.seq_len, data_file='./data/random_example.txt')
    test_dataset = SentenceDataset(seq_len=config.seq_len, data_file='./data/random_example.txt')

    model = GPT(config, vocab_len=len(train_dataset.get_vocab()))
    trainer = Trainer(model, config, train_dataset, train_dataset)
    trainer.train()