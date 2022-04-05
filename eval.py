import torch
import json
import argparse
import glob
import os

from model import GPT, Config
from trainer import EventDataset

parser = argparse.ArgumentParser(description='Eval of the model.')
parser.add_argument('-s','--seed', help='Seed for the model.', required=True)


if __name__ == '__main__':
    args = vars(parser.parse_args())

    config = Config(logging=False)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    train_dataset = EventDataset(seq_len=config.seq_len, data_file='/Users/farleyschaefer/Documents/projects/newco/engine/data/training.json', device=device)
    model = GPT(config, vocab_len=len(train_dataset.get_vocab())).to(device)

    checkpoints = glob.glob('/Users/farleyschaefer/Documents/projects/newco/engine/models/*.pt')
    latest = max(checkpoints, key=os.path.getctime)

    model.load_state_dict(torch.load(latest))
    model.eval()

    seed = args["seed"].split(' ')
    seed_tensor = torch.tensor([train_dataset.events_to_idx(seed)], dtype=torch.long).to(device)
    out, _ = model(seed_tensor, None)
    last_pred = torch.argmax(out[0][-1]).item()
    last_pred = train_dataset.idx_to_event(last_pred)

    print(json.dumps({
        "pred": last_pred
    }))