import gzip
import json
import os

def load_json(filepath, key=None):
    """
    Load a JSON file, automatically detecting if it's gzipped based on the file extension.
    
    Args:
        filepath (str): Path to the JSON file (.json or .json.gz)
        
    Returns:
        dict or list: Parsed JSON data
    """
    if filepath.endswith('.json.gz'):
        with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r', encoding='UTF-8') as f:
            data = json.load(f)
    
    if key is not None:
        data = [d[key] for d in data]
    return data

def save_json(data, filepath, indent=2):
    """
    Save data to a JSON file, automatically handling gzip compression based on the file extension.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath (str): Path to save to (.json or .json.gz)
        indent (int, optional): Number of spaces for indentation. Defaults to 2.
    """
    if filepath.endswith('.json.gz'):
        with gzip.open(filepath, 'wt', encoding='UTF-8') as f:
            json.dump(data, f, indent=indent)
    else:
        with open(filepath, 'w', encoding='UTF-8') as f:
            json.dump(data, f, indent=indent) 
