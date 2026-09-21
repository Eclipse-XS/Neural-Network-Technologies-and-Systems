from ..common.training import run_epoch, fit


def train_one_epoch(model, loader, optimizer, device):
    return run_epoch(model, loader, device, "generation", optimizer=optimizer)


def fit_language_model(model, loaders, vocabulary, device, output_dir,
                       max_epochs=30, patience=5, learning_rate=1e-3, seed=42, dataset_metadata=None):
    return fit(model, loaders, vocabulary, device, output_dir, "generation",
               max_epochs=max_epochs, patience=patience, learning_rate=learning_rate, seed=seed,
               dataset_metadata=dataset_metadata)
