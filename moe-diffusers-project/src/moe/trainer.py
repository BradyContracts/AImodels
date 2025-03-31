class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(self.train_loader)}')

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / len(self.val_loader)}')

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)