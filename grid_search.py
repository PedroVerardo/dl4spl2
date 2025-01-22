import pandas as pd
from utils import LightningModel
import lightning as L
import os
import torch

def select_top_features(df, top_percentage=0.7):
    df_sorted = df.sort_values(by='importance', ascending=False)
    top_n = int(len(df_sorted) * top_percentage)
    top_features = df_sorted.head(top_n)['features'].tolist()
    return df[top_features]

if __name__ == '__main__':
    decision_tree = pd.read_csv('feature_importance_DT.csv')
    random_forest = pd.read_csv('feature_importance_RF.csv')
    gradient_boosting = pd.read_csv('feature_importance_GB.csv')

    top_percentage=0.7

    top_decision_tree = select_top_features(decision_tree, top_percentage)
    top_random_forest = select_top_features(random_forest, top_percentage)
    top_gradient_boosting = select_top_features(gradient_boosting, top_percentage)

    data = pd.read_parquet('data.parquet')
    selected_data = data[top_decision_tree]

    LightningModel(num_features=len(selected_data.columns), activation="PReLU", optimizer_name="Adam", loss_name="MSELoss")

    model = LightningModel(num_features=len(selected_data.columns), activation="PReLU", optimizer_name="Adam", loss_name="MSELoss")

    train_size = int(0.8 * len(selected_data))
    test_size = len(selected_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(selected_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True  
    )

    checkpoint_path = "checkpoints/best-checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        trainer.fit(model, train_loader, test_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, test_loader)

