import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical
import numpy as np

class MultilayerBidirectionalRNN:
    def __init__(self, input_shape=(14, 256), num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_loader, val_loader, epochs=10, checkpoint_path='best_model.h5'):
        # Define the callback for saving the best model
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

        # Prepare validation data
        x_val, y_val = [], []
        for x_batch, y_batch in val_loader:
            x_val.append(x_batch.numpy())
            y_val.append(to_categorical(y_batch.numpy(), num_classes=self.num_classes))

        x_val = np.vstack(x_val)  # Combine batches into one array
        y_val = np.vstack(y_val)  # Combine batches into one array

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            # Training
            for step, (x_batch, y_batch) in enumerate(train_loader):
                y_batch = to_categorical(y_batch.numpy(), num_classes=self.num_classes)
                loss, accuracy = self.model.train_on_batch(x_batch.numpy(), y_batch)
                print(f"Step {step}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")

            # Perform validation at the end of the epoch
            val_loss, val_acc = self.model.evaluate(x_val, y_val, verbose=0)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

            # Checkpoint callback
            checkpoint.on_epoch_end(epoch, logs={'val_accuracy': val_acc, 'val_loss': val_loss})

    def evaluate(self, data_loader):
        total_loss, total_acc = 0.0, 0.0
        num_batches = 0
        for step, (x_batch, y_batch) in enumerate(data_loader):
            y_batch = to_categorical(y_batch.numpy(), num_classes=self.num_classes)
            loss, acc = self.model.evaluate(x_batch.numpy(), y_batch, verbose=0)
            total_loss += loss
            total_acc += acc
            num_batches += 1
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        return avg_loss, avg_acc

    def predict(self, data_loader):
        predictions = []
        for x_batch, _ in data_loader:
            preds = self.model.predict(x_batch.numpy())
            predictions.extend(preds)
        return np.array(predictions)

