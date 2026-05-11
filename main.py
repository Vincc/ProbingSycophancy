import argparse
import json
import logging
import sys
from pathlib import Path

def extract_mha_activations(model, tokenizer, inputs):
    
