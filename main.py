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
    # model.load_state_dict(torch.load("/Users/farleyschaefer/Desktop/chkpt-15.pt", map_location=device))

    trainer = Trainer(model, config, train_dataset, None)
    trainer.train()

    model.eval()

    seed = "this is the work that we want".lower().split(' ')
    print(f'Seed: {" ".join(seed)}')

    for _ in range(10):
        seed_tensor = torch.tensor([train_dataset.words_to_idx(seed)], dtype=torch.long).to(device)
        out, _ = model(seed_tensor, None)
        last_pred = torch.argmax(out[0][-1]).item()
        last_pred = train_dataset.idx_to_word(last_pred)
        seed.append(last_pred)

    print(f'Output: {" ".join(seed)}')
