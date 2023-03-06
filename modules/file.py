from config import *

import torch


def saveWeights(model_state_dict, optimizer_state_dict, best_acc, current_acc):
    torch.save(model_state_dict, f"{WEIGHT_FILENAME}")
    print(f"Saved best model state to {WEIGHT_FILENAME}")
    torch.save(optimizer_state_dict, f"{WEIGHT_FILENAME}o")
    print(f"Saved best optimizer state to {WEIGHT_FILENAME}o")

    print(f"Valid acc: {best_acc:>2.5f} -> {current_acc:>2.5f}\n")


def saveEpochInfo(epoch, train_acc, valid_acc, train_loss, valid_loss):
    with open(f"{WEIGHT_FILENAME}_info.log", "w") as f:
        f.write(f"Epoch: {epoch+1}\n")
        f.write(f"Train acc: {train_acc * 100:>2.5f}%\n")
        f.write(f"Valid acc: {valid_acc * 100:>2.5f}%\n")
        f.write(f"Train loss: {train_loss:>2.5f}\n")
        f.write(f"Valid loss: {valid_loss:>2.5f}\n")

