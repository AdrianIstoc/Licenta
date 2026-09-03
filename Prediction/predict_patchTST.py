import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os

CONTEXT = 54
PATCH = 12
STRIDE = 3
LR = 0.0003

class PatchtTST(nn.Module):
    def __init__(
        self, 
        context_length=CONTEXT, 
        patch_length=PATCH, 
        patch_stride=STRIDE, 
        prediction_length=6,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.context_length = context_length
        self.patch_length=patch_length
        self.patch_stride=patch_stride
        self.prediction_length=prediction_length

        self.num_patches = ((context_length-patch_length)// patch_stride)+1
        self.patch__embedding = nn.Linear(patch_length, d_model)
        self.positional_embedding=nn.Parameter(torch.randn(1, self.num_patches,d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
        self.head = nn.Linear(
            self.num_patches*d_model,
            prediction_length
        )

    def create_patches(self, x):
        patches = x.unfold(
            dimension = 1,
            size=self.patch_length,
            step=self.patch_stride
        )
        return patches

    def forward(self, x):
        patches = self.create_patches(x)

        x=self.patch__embedding(patches)
        x=x+self.positional_embedding
        x=self.encoder(x)
        x=x.reshape(x.shape[0], -1)

        prediction=self.head(x)

        return prediction

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_patchtst_dataset(values, context_length=CONTEXT, horizon=6):
    values=np.asarray(values, dtype=np.float32)

    x=[]
    y=[]

    total_length = context_length + horizon

    if len(values) < total_length:
        return None, None

    for start in range(len(values) - total_length+1):
        context = values[start:start +context_length]

        target = values[start+context_length:start+context_length+horizon]

        x.append(context)
        y.append(target)

    x=torch.tensor(np.asarray(x),dtype=torch.float32)
    y=torch.tensor(np.asarray(y),dtype=torch.float32)

    return x, y

def train_patchtst(
        values,
        context_length=CONTEXT,
        patch_length=PATCH,
        patch_stride=STRIDE,
        horizon=6,
        epochs=50,
        learning_rate=LR,
        batch_size=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
):
    set_seed(42)
    
    values = np.asarray(values, dtype=np.float32)

    mean=values.mean()
    std=values.std()

    if std==0:
        return None

    scaled_values=(values-mean)/std

    x, y = create_patchtst_dataset(values=scaled_values, context_length=context_length, horizon=horizon)

    if x is None:
        return None

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model = PatchtTST(
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        prediction_length=horizon,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    dataset = torch.utils.data.TensorDataset(x,y)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            prediction=model(batch_x)

            loss=criterion(prediction, batch_y)

            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        
        if(epoch +1)%10==0:
            average_loss=(total_loss/len(loader))

            print(f"Epoch {epoch+1}/{epochs} "
                    f"loss={average_loss:.6f}")

    return model, mean, std

def forecast_patchtst(model, values, mean, std, context_length=CONTEXT):
    values = np.asarray(values, dtype = np.float32)

    if len(values)<context_length:
        return None

    context = values[-context_length:]

    scaled_context = (context - mean) / std

    x = torch.tensor(scaled_context, dtype=torch.float32).unsqueeze(0)

    device = next(model.parameters()).device
    x= x.to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(x)

    prediction=(prediction.squeeze(0).cpu().numpy())

    prediction = prediction*std + mean
    return prediction

def predict_patchtst(df, n, context_length=CONTEXT, patch_length=PATCH, patch_stride=STRIDE, epochs=50, learning_rate=LR, batch_size=16, d_model=64, n_heads=4, n_layers=2, dropout=0.1, model_path=None):
    df = df.copy()
    df = df.sort_values("date")

    if model_path is None:
        model_path = (
            f"models/patchtst_value_"
            f"c{context_length}_"
            f"p{patch_length}_"
            f"s{patch_stride}_"
            f"lr{learning_rate}_"
            f"h{n}_"
            f"dm{d_model}_"
            f"nh{n_heads}_"
            f"nl{n_layers}_"
            f"do{dropout}.pth"
        )

    valid = df[df["value"].notna()].copy()

    if valid.empty:
        return None

    values = valid["value"].to_numpy(dtype=np.float32)

    if len(values) < context_length+n:
        return None

    if os.path.exists(model_path):
        loaded = load_patchtst(model_path,prediction_length=n)
        if loaded is not None:
            value_model, value_mean, value_std = loaded
        else:
            print("Se antreneaza value...")
            value_results=train_patchtst(values=values, context_length=context_length, patch_length=patch_length, patch_stride=patch_stride,horizon=n,epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)

            if value_results is None:
                return None

            value_model, value_mean, value_std = value_results

            save_patchtst(model=value_model,mean=value_mean,std=value_std,path=model_path,context_length=context_length,patch_length=patch_length,patch_stride=patch_stride,prediction_length=n,d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)
    else:
        print("Se antreneaza value...")

        value_model, value_mean, value_std = train_patchtst(
            values=values,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            horizon=n,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )

        if value_model is None:
            return None

        save_patchtst(model=value_model,mean=value_mean,std=value_std,path=model_path,context_length=context_length, patch_length=patch_length, patch_stride=patch_stride,prediction_length=n,d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)


    value_prediction = forecast_patchtst(model=value_model, values=values,mean=value_mean,std=value_std, context_length=context_length)

    if value_prediction is None:
        return None

    percentage_prediction = None

    if "percentage" in df.columns:
        percentage_values=(df["percentage"].dropna().to_numpy(dtype=np.float32))


        if len(percentage_values) >= context_length+n:
            percentage_model_path = (
                f"models/patchtst_percentage_"
                f"c{context_length}_"
                f"p{patch_length}_"
                f"s{patch_stride}_"
                f"lr{learning_rate}_"
                f"h{n}_"
                f"dm{d_model}_"
                f"nh{n_heads}_"
                f"nl{n_layers}_"
                f"do{dropout}.pth"
            )

            if os.path.exists(percentage_model_path):
                loaded = load_patchtst(percentage_model_path, prediction_length=n)

                if loaded is not None:
                    percentage_model, percentage_mean, percentage_std = loaded
                else:
                    print("Se antreneaza percentage...")
                    percentage_results=train_patchtst(values=percentage_values,context_length=context_length,patch_length=patch_length,patch_stride=patch_stride,horizon=n,epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)

                    if percentage_results is None:
                        percentage_model = None
                    else:
                        percentage_model, percentage_mean,percentage_std = percentage_results
                        save_patchtst(model=percentage_model,mean=percentage_mean,std=percentage_std, path=percentage_model_path,context_length=context_length,patch_length=patch_length,patch_stride=patch_stride,prediction_length=n,d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)
            else:
                print("Se antreneaza percentage...")

                percentage_results = train_patchtst(values=percentage_values,context_length=context_length, patch_length=patch_length, patch_stride=patch_stride,horizon=n, epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)
                
                if percentage_results is not None:
                    percentage_model, percentage_mean, percentage_std = percentage_results

                    save_patchtst(model=percentage_model, mean=percentage_mean, std=percentage_std, path=percentage_model_path, context_length=context_length, patch_length=patch_length, patch_stride=patch_stride,prediction_length=n, d_model=d_model,n_heads=n_heads,n_layers=n_layers,dropout=dropout)
                else:
                    percentage_model = None

            if percentage_model is not None:
                percentage_prediction=forecast_patchtst(model=percentage_model, values=percentage_values, mean=percentage_mean,std=percentage_std,context_length=context_length)
                if percentage_prediction is not None:
                    percentage_prediction =np.clip(percentage_prediction,0,100)


    last_real_date = valid["date"].max()

    predictions = []

    for i in range(n):

        next_date = last_real_date + pd.DateOffset(months=i+1)

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "value": float(value_prediction[i]),
            "percentage": (float(percentage_prediction[i]) if percentage_prediction is not None else None)
        })

    return predictions




def save_patchtst(
        model,
        mean,
        std,
        path,
        context_length,
        patch_length,
        patch_stride,
        prediction_length,
        d_model,
        n_heads,
        n_layers,
        dropout
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),

        "mean": mean,
        "std": std,

        "context_length": context_length,
        "patch_length": patch_length,
        "patch_stride": patch_stride,
        "prediction_length": prediction_length,

        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout
    }

    torch.save(checkpoint, path)


def load_patchtst(path, prediction_length=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    saved_prediction_length = checkpoint["prediction_length"]

    if (prediction_length is not None and saved_prediction_length != prediction_length):
        print("Se v-a antrena un model nou.")
        return None

    model = PatchtTST(
        context_length=checkpoint["context_length"],
        patch_length=checkpoint["patch_length"],
        patch_stride=checkpoint["patch_stride"],
        prediction_length=checkpoint["prediction_length"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std