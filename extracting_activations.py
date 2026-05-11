import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

def extract_mha_activations(model, inputs):
    model.eval()
    