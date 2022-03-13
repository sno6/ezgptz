from model import GPT, GPTConfig
from trainer import MathDataset, Trainer, TrainerConfig

import wandb
wandb.init(project="ezgptz", entity="sno6")

if __name__ == '__main__':
    train_dataset = MathDataset()
    trainer_cfg = TrainerConfig()
    model_cfg = GPTConfig(vocab_len=train_dataset.vocab_len())
    model = GPT(model_cfg)

    wandb.config = {
        "learning_rate": trainer_cfg.learning_rate,
        "epochs": trainer_cfg.epochs,
        "batch_size": trainer_cfg.batch_size,
    }

    trainer = Trainer(model, train_dataset, train_dataset, trainer_cfg)
    trainer.train()