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

    def train(self, x_train, y_train, batch_size=1024, epochs=10, validation_data=None, checkpoint_path="best_model.h5"):
        """
        Trains the model on the provided training data.
        
        Parameters:
        - x_train: Training input data.
        - y_train: Training labels (one-hot encoded).
        - batch_size: Size of the batches used for training.
        - epochs: Number of training epochs.
        - validation_data: Tuple (x_val, y_val) for validation during training.
        
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
        
        # Training the model
        history = self.model.fit(x_train, y_train, 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                validation_data=validation_data,
                                callbacks=callbacks)
        
        return history

    def evaluate(self, x_test, y_test):
        """
        Evaluates the model on the test data.
        
        Parameters:
        - x_test: Test input data.
        - y_test: Test labels (one-hot encoded).
        
        Returns:
        - Loss and accuracy of the model on the test set.
        """
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_input):
        """
        Generates predictions for the input data.
        
        Parameters:
        - x_input: Input data for which predictions are to be made.
        
        Returns:
        - Predicted classes.
        """
        return self.model.predict(x_input)