import argparse
#import sys
#sys.path.append('Desktop/MLops/mlops_cc_project/src/data')

import importlib.util

spec = importlib.util.spec_from_file_location("make_dataset", "/src/data/make_dataset.py")
make_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_dataset)

import matplotlib.pyplot as plt
import torch
from make_dataset import CorruptMnist
from model import MyAwesomeModel




class Train(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for training",
            usage="python train_model.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=1e-3)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        model = model.to(self.device)
        train_set = CorruptMnist(train=True)
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        n_epoch = 5
        for epoch in range(n_epoch):
            loss_tracker = []
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                preds = model(x.to(self.device))
                loss = criterion(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
                loss_tracker.append(loss.item())
            print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
        torch.save(model.state_dict(), "models/trained_model.pt")

        plt.plot(loss_tracker, "-")
        plt.xlabel("Training step")
        plt.ylabel("Training loss")
        plt.savefig("/reports/figures/training_curve.png")

        return model



# I am writing this line because I want to have a line with lenth > 79 chracters but < 100
if __name__ == "__main__":
    Train()
