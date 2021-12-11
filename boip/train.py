from tqdm import tqdm


def fit_model(train_X, model, optimizer, criterion, epochs=150, verbose=True):
    model.train()

    for _ in tqdm(range(epochs), 'Training', leave=False, unit='epoch', disable=not verbose):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -criterion(output, model.train_targets)
        loss.backward()
        optimizer.step()
    
