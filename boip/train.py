from tqdm import tqdm


def fit_model(train_X, model, optimizer, criterion, epochs=150, verbose: int = 0):
    model.train()

    for _ in tqdm(range(epochs), "Training", leave=False, unit="epoch", disable=verbose < 2):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -criterion(output, model.train_targets)
        loss.backward()
        optimizer.step()
