from keras.layers import Input, Bidirectional, LSTM, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

class MultilayerBidirectionalRNN:
    def __init__(self, input_shape=(14, 256), num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the multilayer bidirectional RNN model according to the given architecture.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Bidirectional Layer 1
        x = Bidirectional(LSTM(256, return_sequences=True))(inputs)

        # Bidirectional Layer 2
        x = Bidirectional(LSTM(128, return_sequences=True))(x)

        # Bidirectional Layer 3
        x = Bidirectional(LSTM(64, return_sequences=False))(x)

        # Flattening the output
        x = Flatten()(x)

        # Dense layer with softmax for classification (10 classes)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def summary(self):
        """
        Prints the summary of the model architecture.
        """
        return self.model.summary()

    def train(self, train_loader, batch_size=1024, epochs=10, validation_loader=None, checkpoint_path="best_model.keras"):
        """
        Trains the model on the provided training data.
        
        Parameters:
        - train_loader: Training data loader.
        - batch_size: Size of the batches used for training.
        - epochs: Number of training epochs.
        - validation_loader: Data loader for validation data (optional).
        
        Returns:
        - History of the training process.
        """
        # Create a callback that saves the model's best weights
        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_loss',  # Monitors the validation loss
                                                        save_best_only=True,  # Saves only the best model
                                                        mode='min',  # Indicates that lower is better for val_loss
                                                        verbose=1)
        
        # List of callbacks
        callbacks = [checkpoint]
        
        # If a validation_loader is provided, convert it to a format Keras can use
        if validation_loader is not None:
            x_val, y_val = next(iter(validation_loader))  # Assuming single batch validation, you can modify this to suit your needs
            validation_data = (x_val, y_val)
        else:
            validation_data = None

        # Training the model
        history = self.model.fit(train_loader, 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                validation_data=validation_data,
                                callbacks=callbacks)
        
        return history

    def evaluate(self, test_loader, batch_size=1024):
        """
        Evaluates the model on the provided test data.

        Parameters:
        - test_loader: Data loader for the test data.
        - batch_size: Size of the batches used for evaluation.

        Returns:
        - Evaluation metrics (loss and accuracy, etc.).
        """
        x_test, y_test = next(iter(test_loader))  # Assuming a single batch of test data
        # Evaluating the model
        results = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

        return results


    def predict(self, data_loader, batch_size=1024):
        """
        Generates predictions using the trained model.

        Parameters:
        - data_loader: Data loader containing the input data.
        - batch_size: Size of the batches used for predictions.

        Returns:
        - Model predictions.
        """
        x_data = next(iter(data_loader))  # Assuming a single batch of data
        # Predicting the outputs
        predictions = self.model.predict(x_data, batch_size=batch_size, verbose=1)

        return predictions