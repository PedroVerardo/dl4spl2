import lightning as L
from torch import nn

class LightningModel(L.LightningModule):
    def __init__(self, num_features, activation="ReLU", optimizer_name="Adam", loss_name="MSELoss"):
        super().__init__()
        self.num_features = num_features
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.model = self.build_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self(x)
        loss = self.get_loss_function()(z, x)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self(x)
        loss = self.get_loss_function()(z, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.get_optimizer()

    def get_optimizer(self):
        optimizers = {
            "Adam": torch.optim.Adam(self.parameters(), lr=0.001),
            "SGD": torch.optim.SGD(self.parameters(), lr=0.001),
            "RMSprop": torch.optim.RMSprop(self.parameters(), lr=0.001)
        }
        return optimizers[self.optimizer_name]

    def get_loss_function(self):
        loss_functions = {
            "MSELoss": nn.MSELoss(),
            "L1Loss": nn.L1Loss(),
            "SmoothL1Loss": nn.SmoothL1Loss(),
            "CrossEntropyLoss": nn.CrossEntropyLoss()
        }
        return loss_functions[self.loss_name]

    def get_activation(self):
        activations = {
            "ReLU": nn.ReLU(),
            "PReLU": nn.PReLU(),
            "ELU": nn.ELU()
        }
        return activations[self.activation]

    def build_model(self):
        hidden_size = self.num_features // 2
        hidden_size2 = hidden_size // 2
        return nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            self.get_activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size2),
            self.get_activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, 1)
        )