from ..common.training import run_epoch, fit


def train_one_epoch(model, loader, optimizer, device, weights):
    return run_epoch(model, loader, device, "classification", weights, optimizer)


def fit_classifier(model, loaders, vocabulary, device, output_dir, weights,
                   max_epochs=30, patience=5, learning_rate=1e-3, seed=42):
    return fit(model, loaders, vocabulary, device, output_dir, "classification", weights,
               max_epochs=max_epochs, patience=patience, learning_rate=learning_rate, seed=seed)
