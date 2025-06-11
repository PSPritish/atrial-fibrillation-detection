def calculate_accuracy(y_true, y_pred):
    correct_predictions = (y_true == y_pred).sum().item()
    total_predictions = y_true.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_loss(loss_fn, y_true, y_pred):
    return loss_fn(y_pred, y_true)

def log_metrics(metrics_dict, step):
    for key, value in metrics_dict.items():
        print(f'Step {step}: {key} = {value:.4f}')

def calculate_complex_metrics(y_true, y_pred):
    # Placeholder for complex-valued metrics calculation
    # Implement specific metrics for complex-valued outputs if needed
    return {}  # Return a dictionary of complex metrics

def track_training_metrics(loss, accuracy, step):
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    log_metrics(metrics, step)