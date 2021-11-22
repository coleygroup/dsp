from tqdm import trange


def fit_model(train_X, model, optimizer, criterion, epochs=150):
    model.train()

    for _ in trange(epochs, desc='Training', leave=False, unit='epoch', disable=True):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -criterion(output, model.train_targets)
        loss.backward()
        optimizer.step()