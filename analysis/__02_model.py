import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Concatenate, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt

def train(X, Y, dataset_type, num_classes=10, epochs=500, batch_size=128, type='FullBand', channels='two_channel'):
    """
    Train a model for Envisioned Speech Recognition and evaluate its final performance.

    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_timepoints, n_channels).
    Y (numpy.ndarray): Labels corresponding to each sample.
    dataset_type (str): Type of dataset, e.g., 'digit', 'alphabet', or 'object'.
    num_classes (int): Number of output classes for classification.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    Model: Trained Keras model.
    """
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Ensure the input shape matches the selected channels
    input_shape = (X.shape[1], X.shape[2])  # No need to filter channels again
    i1 = Input(shape=input_shape)
    x1 = BatchNormalization()(i1)
    x1 = Conv1D(128, kernel_size=10, strides=1, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = LSTM(256, activation='tanh')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(128, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    output = Dense(num_classes, activation='softmax')(x1)
    model = Model(inputs=i1, outputs=output)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Define early stopping and model checkpoint
    es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)
    checkpoint_path = f"model/{channels}/model_{dataset_type}_{type}.keras"
    mc = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(X_train, y=to_categorical(Y_train, num_classes=num_classes),
                        validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        verbose=1, callbacks=[es, mc])

    # Load the best model from the checkpoint
    best_model = load_model(checkpoint_path)

    # Evaluate the model on the test set
    test_loss, test_accuracy = best_model.evaluate(X_test, to_categorical(Y_test, num_classes=num_classes), verbose=1)
    print(f"\nFinal Test Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Plot training and validation accuracy and loss
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return best_model, X_test, Y_test




def train_LateFusion(X_GAMMA, X_BETA, Y, selected_channels, dataset_type, num_classes=10, epochs=500, batch_size=128):
    """
    Train two separate models (GAMMA and BETA) and apply late fusion.

    Parameters:
    X_GAMMA (numpy.ndarray): GAMMA wave feature matrix.
    X_BETA (numpy.ndarray): BETA wave feature matrix.
    Y (numpy.ndarray): Labels corresponding to each sample.
    selected_channels (list): List of selected channels for input shape.
    dataset_type (str): Type of dataset, e.g., 'digit', 'alphabet', or 'object'.
    num_classes (int): Number of output classes.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    float: Late fusion accuracy.
    """
    # Train the GAMMA model
    model_GAMMA, X_test_GAMMA, Y_test = train(
        X_GAMMA, Y, selected_channels, dataset_type, num_classes, epochs, batch_size, type='GAMMA'
    )

    # Train the BETA model
    model_BETA, X_test_BETA, _ = train(
        X_BETA, Y, selected_channels, dataset_type, num_classes, epochs, batch_size, type='BETA'
    )

    # Get predictions from both models
    gamma_preds = model_GAMMA.predict(X_test_GAMMA)
    beta_preds = model_BETA.predict(X_test_BETA)

    # Evaluate individual models
    Y_pred_GAMMA = np.argmax(gamma_preds, axis=1)
    gamma_accuracy = accuracy_score(Y_test, Y_pred_GAMMA)
    print(f"GAMMA Model Accuracy: {gamma_accuracy * 100:.2f}%")

    Y_pred_BETA = np.argmax(beta_preds, axis=1)
    beta_accuracy = accuracy_score(Y_test, Y_pred_BETA)
    print(f"BETA Model Accuracy: {beta_accuracy * 100:.2f}%")

    # Late Fusion
    combined_preds = (gamma_preds + beta_preds) / 2
    Y_combined_pred = np.argmax(combined_preds, axis=1)

    # Calculate Late Fusion Accuracy
    late_fusion_accuracy = accuracy_score(Y_test, Y_combined_pred)
    print(f"Late Fusion Accuracy: {late_fusion_accuracy * 100:.2f}%")

    # Plot confusion matrix for late fusion
    sns.heatmap(confusion_matrix(Y_test, Y_combined_pred), annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix - Late Fusion')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return gamma_accuracy, beta_accuracy, late_fusion_accuracy

def visualize(model, X, Y, num_classes=10, perplexity=30, n_iter=1000, dataset_type=None):
    """
    Visualize the test dataset using t-SNE based on features extracted from a trained model.

    Parameters:
    model (Model): Trained Keras model.
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_timepoints, n_channels).
    Y (numpy.ndarray): Labels corresponding to each sample.
    selected_channels (list): List of channel indices to use.
    num_classes (int): Number of output classes for classification.
    perplexity (int): Perplexity parameter for t-SNE.
    n_iter (int): Number of iterations for t-SNE optimization.
    dataset_type (str): Name of the dataset type, e.g., 'alphabet', 'digit'.

    Returns:
    None: Displays a t-SNE plot.
    """
    # Split the data (No need to reapply selected_channels here)
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Extract intermediate features
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = intermediate_layer_model.predict(X_test)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(features)

    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_test, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Labels')
    plt.title(f't-SNE Visualization of {dataset_type} Test Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

def train_multibranch(X_BETA, X_GAMMA, Y, num_classes=10, epochs=500, batch_size=128, selected_channels=None):
    """
    Train a multi-branch model for classification using BETA and GAMMA wave data with channel selection.
    
    Parameters:
    X_BETA (numpy.ndarray): Feature matrix for BETA waves, shape (n_samples, n_timepoints, n_channels).
    X_GAMMA (numpy.ndarray): Feature matrix for GAMMA waves, shape (n_samples, n_timepoints, n_channels).
    Y (numpy.ndarray): Labels corresponding to each sample.
    num_classes (int): Number of output classes for classification.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    selected_channels (list of int): List of selected channel indices to use.
    
    Returns:
    Model: Trained Keras multi-branch model.
    """
    
    # Select specified channels if provided
    if selected_channels is not None:
        X_BETA = X_BETA[:, :, selected_channels]
        X_GAMMA = X_GAMMA[:, :, selected_channels]
    
    input_shape = (X_BETA.shape[1], X_BETA.shape[2])
    
    # BETA Branch
    input_beta = Input(shape=input_shape, name="BETA_Input")
    beta_branch = BatchNormalization()(input_beta)
    beta_branch = Conv1D(128, kernel_size=10, strides=1, activation='relu', padding='same')(beta_branch)
    beta_branch = BatchNormalization()(beta_branch)
    beta_branch = MaxPooling1D(2)(beta_branch)
    beta_branch = LSTM(256, activation='tanh')(beta_branch)
    beta_branch = BatchNormalization()(beta_branch)
    beta_branch = Dense(128, activation='relu')(beta_branch)
    beta_branch = Dropout(0.5)(beta_branch)
    
    # GAMMA Branch
    input_gamma = Input(shape=input_shape, name="GAMMA_Input")
    gamma_branch = BatchNormalization()(input_gamma)
    gamma_branch = Conv1D(128, kernel_size=10, strides=1, activation='relu', padding='same')(gamma_branch)
    gamma_branch = BatchNormalization()(gamma_branch)
    gamma_branch = MaxPooling1D(2)(gamma_branch)
    gamma_branch = LSTM(256, activation='tanh')(gamma_branch)
    gamma_branch = BatchNormalization()(gamma_branch)
    gamma_branch = Dense(128, activation='relu')(gamma_branch)
    gamma_branch = Dropout(0.5)(gamma_branch)
    
    # Combine the branches
    combined = Concatenate()([beta_branch, gamma_branch])
    
    # Additional dense layers after combining
    combined = Dense(256, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name="Output")(combined)
    
    # Build and compile the model
    model = Model(inputs=[input_beta, input_gamma], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Split the data into training and test sets
    X_beta_train, X_beta_test, X_gamma_train, X_gamma_test, Y_train, Y_test = train_test_split(
        X_BETA, X_GAMMA, Y, test_size=0.2, random_state=1)
    
    # Define early stopping and model checkpoint
    es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)
    mc = ModelCheckpoint('multi_branch_model.keras', save_best_only=True, verbose=1)
    
    # Train the model
    model.fit(
        [X_beta_train, X_gamma_train],
        to_categorical(Y_train, num_classes=num_classes),
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es, mc]
    )
    
    # Evaluate the model
    pred = model.predict([X_beta_test, X_gamma_test])
    Y_pred = np.argmax(pred, axis=1)
    print("Accuracy:", accuracy_score(Y_pred, Y_test))
    
    # Plot confusion matrix
    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.show()
    
    return model
       

def evaluate(model, X, Y, selected_channels, dataset_type, num_classes=10, batch_size=128, type='FullBand', channels='two_channel'):
    """
    Evaluate a pre-trained model for Envisioned Speech Recognition.

    Parameters:
    model (Model): Pre-trained Keras model.
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_timepoints, n_channels).
    Y (numpy.ndarray): Labels corresponding to each sample.
    selected_channels (list): List of channel indices to use (only for input shape).
    dataset_type (str): Type of dataset, e.g., 'digit', 'alphabet', or 'object'.
    num_classes (int): Number of output classes for classification.
    batch_size (int): Batch size for evaluation.

    Returns:
    float: Accuracy of the model on the test set.
    """
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Evaluate the model
    pred = model.predict(X_test, batch_size=batch_size)
    Y_pred = np.argmax(pred, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix
    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return accuracy


def attention_mechanism(gamma_preds, beta_preds):
    """
    Apply an attention mechanism to dynamically weight GAMMA and BETA predictions.

    Parameters:
    gamma_preds (Tensor): Output from model_gamma.
    beta_preds (Tensor): Output from model_beta.

    Returns:
    Tensor: Weighted combination of gamma_preds and beta_preds.
    """
    # Concatenate predictions from both models
    combined_inputs = Concatenate()([gamma_preds, beta_preds])

    # Attention layer to generate weights for each model's output
    attention_weights = Dense(2, activation='softmax', name='attention_weights')(combined_inputs)

    # Split the attention weights
    gamma_weight = attention_weights[:, 0:1]
    beta_weight = attention_weights[:, 1:2]

    # Apply attention weights to the respective model outputs
    weighted_gamma = Multiply()([gamma_preds, gamma_weight])
    weighted_beta = Multiply()([beta_preds, beta_weight])

    # Combine weighted outputs
    combined_outputs = Concatenate()([weighted_gamma, weighted_beta])

    return combined_outputs

def build_attention_fusion_model(model_GAMMA, model_BETA, num_classes=10):
    """
    Build an attention-based late fusion model.

    Parameters:
    model_GAMMA (Model): Pre-trained GAMMA model.
    model_BETA (Model): Pre-trained BETA model.
    num_classes (int): Number of output classes.

    Returns:
    Model: Attention-based fusion model.
    """
    # Inputs for both GAMMA and BETA
    input_gamma = model_GAMMA.input
    input_beta = model_BETA.input

    # Outputs of pre-trained models
    gamma_preds = model_GAMMA.output
    beta_preds = model_BETA.output

    # Apply attention mechanism
    fused_features = attention_mechanism(gamma_preds, beta_preds)

    # Final classification layer
    output = Dense(num_classes, activation='softmax', name='final_output')(fused_features)

    # Build and compile the model
    fusion_model = Model(inputs=[input_gamma, input_beta], outputs=output)
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return fusion_model

def train_Attention_LateFusion(X_GAMMA, X_BETA, Y, selected_channels, dataset_type, num_classes=10, epochs=500, batch_size=128):
    """
    Train two separate models (GAMMA and BETA) and apply attention-based late fusion.

    Parameters:
    X_GAMMA (numpy.ndarray): GAMMA wave feature matrix.
    X_BETA (numpy.ndarray): BETA wave feature matrix.
    Y (numpy.ndarray): Labels corresponding to each sample.
    selected_channels (list): List of selected channels for input shape.
    dataset_type (str): Type of dataset, e.g., 'digit', 'alphabet', or 'object'.
    num_classes (int): Number of output classes.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    float: Attention-based late fusion accuracy.
    """
    # Train the GAMMA model
    model_GAMMA, X_test_GAMMA, Y_test = train(
        X_GAMMA, Y, selected_channels, dataset_type, num_classes, epochs, batch_size, type='GAMMA'
    )

    # Train the BETA model
    model_BETA, X_test_BETA, _ = train(
        X_BETA, Y, selected_channels, dataset_type, num_classes, epochs, batch_size, type='BETA'
    )

    # Build attention-based fusion model
    fusion_model = build_attention_fusion_model(model_GAMMA, model_BETA, num_classes)

    # Train the fusion model
    early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=10,             # Stop after 10 epochs of no improvement
    verbose=1,               # Print stopping logs
    restore_best_weights=True  # Restore the best weights after stopping
)

# Fit the model with EarlyStopping
    fusion_model.fit(
        [X_test_GAMMA, X_test_BETA],
        to_categorical(Y_test, num_classes=num_classes),
        epochs=500,                # Maximum number of epochs
        batch_size=batch_size,
        validation_split=0.2,     # Use 20% of training data for validation
        verbose=1,
        callbacks=[early_stopping]  # Include EarlyStopping in callbacks
    )

    # Evaluate the fusion model
    preds = fusion_model.predict([X_test_GAMMA, X_test_BETA])
    Y_pred = np.argmax(preds, axis=1)
    attention_fusion_accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Attention-Based Late Fusion Accuracy: {attention_fusion_accuracy * 100:.2f}%")

    # Plot confusion matrix for attention-based fusion
    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix - Attention Late Fusion')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return attention_fusion_accuracy
