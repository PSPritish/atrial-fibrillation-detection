def save_model(model, filepath):
    """Save the trained model to the specified filepath."""
    import torch
    torch.save(model.state_dict(), filepath)

def load_model(model_class, filepath):
    """Load the model state from the specified filepath."""
    import torch
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    return model

def save_results(results, filepath):
    """Save experiment results to a file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f)

def load_results(filepath):
    """Load experiment results from a file."""
    import json
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results

def create_directory(directory):
    """Create a directory if it does not exist."""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)