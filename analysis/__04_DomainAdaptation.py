import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_shape):
        super(Generator, self).__init__()
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, np.prod(output_shape)),
        )

    def forward(self, x):
        flat_output = self.model(x)
        return flat_output.view(-1, *self.output_shape)


class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(Discriminator, self).__init__()
        self.input_dim = np.prod(input_shape)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def preprocess_data(X, Y, target_class):
    """
    Filter the dataset for a specific class and flatten the input data.
    """
    indices = np.where(Y == target_class)[0]
    X_filtered = X[indices]
    Y_filtered = Y[indices]
    X_flattened = X_filtered.reshape(X_filtered.shape[0], -1)
    return torch.tensor(X_flattened, dtype=torch.float32), torch.tensor(Y_filtered, dtype=torch.float32)


def train_gan(vie_data, emotiv_data, generator, discriminator, epochs=2000, batch_size=32, lr=0.0002):
    vie_loader = DataLoader(TensorDataset(vie_data), batch_size=batch_size, shuffle=True, drop_last=True)
    emotiv_loader = DataLoader(TensorDataset(emotiv_data), batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Loss function
    criterion = torch.nn.BCELoss()

    best_g_loss = float('inf')

    for epoch in tqdm(range(epochs), desc="Training GAN"):
        for (vie_batch,), (emotiv_batch,) in zip(vie_loader, emotiv_loader):
            # Flatten input for the generator
            vie_batch = vie_batch.view(vie_batch.size(0), -1)

            # Train discriminator
            discriminator.zero_grad()
            real_labels = torch.ones(emotiv_batch.size(0), 1)
            real_output = discriminator(emotiv_batch)
            real_loss = criterion(real_output, real_labels)

            fake_data = generator(vie_batch)
            fake_labels = torch.zeros(fake_data.size(0), 1)
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            generator.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Update best loss
            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()

    return generator


def save_model(generator, class_label, save_dir="domainAdaptation_models"):
    """
    Save the trained generator for a specific class.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"generator_class_{class_label}.pth")
    torch.save(generator.state_dict(), model_path)


def load_model(class_label, input_dim, hidden_dim, output_shape, save_dir="domainAdaptation_models"):
    """
    Load a saved generator for a specific class.
    """
    generator = Generator(input_dim=input_dim, hidden_dim=hidden_dim, output_shape=output_shape)
    model_path = os.path.join(save_dir, f"generator_class_{class_label}.pth")
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    return generator


def class_conditional_domain_adaptation(X_VIE, Y_VIE, X_EMOTIV, Y_EMOTIV, epochs=2000, batch_size=32, hidden_dim=512, lr=0.0002, save_dir="domainAdaptation_models"):
    unique_classes = np.unique(Y_VIE)
    input_dim = X_VIE.shape[1] * X_VIE.shape[2]
    output_shape = (X_VIE.shape[1], X_VIE.shape[2])

    X_VIE_transformed = np.zeros_like(X_VIE)

    for class_label in tqdm(unique_classes, desc="Class-Conditional Adaptation"):
        # Filter data for the current class
        vie_data, _ = preprocess_data(X_VIE, Y_VIE, class_label)
        emotiv_data, _ = preprocess_data(X_EMOTIV, Y_EMOTIV, class_label)

        # Initialize generator and discriminator
        generator = Generator(input_dim=input_dim, hidden_dim=hidden_dim, output_shape=output_shape)
        discriminator = Discriminator(input_shape=output_shape, hidden_dim=hidden_dim)

        # Train GAN for this class
        trained_generator = train_gan(vie_data, emotiv_data, generator, discriminator, epochs=epochs, batch_size=batch_size, lr=lr)

        # Save the trained generator
        save_model(trained_generator, class_label, save_dir)

        # Transform VIE data for this class
        transformed_vie_data = trained_generator(vie_data).detach().numpy()
        indices = np.where(Y_VIE == class_label)[0]
        X_VIE_transformed[indices] = transformed_vie_data.reshape(len(indices), *output_shape)

    return X_VIE_transformed
