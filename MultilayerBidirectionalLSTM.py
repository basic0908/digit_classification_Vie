from keras.layers import Input, Bidirectional, LSTM, Dense, Flatten, ELU, Softmax
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np


class MultilayerBidirectionalLSTM:
    def __init__(self, input_shape=(14, 256), num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def build_model(input_shape, num_classes):
        """
        Build the LSTM model based on the EPOC architecture.
        :param input_shape: Shape of the input data (timesteps, features)
        :param num_classes: Number of output classes
        :return: Keras model
        """
        input_layer = Input(shape=input_shape)

        # First Bidirectional LSTM Layer with 256 units
        x = Bidirectional(LSTM(256, return_sequences=True))(input_layer)

        # Second Bidirectional LSTM Layer with 128 units
        x = Bidirectional(LSTM(128, return_sequences=True))(x)

        # Third Bidirectional LSTM Layer with 64 units
        x = Bidirectional(LSTM(64))(x)  # return_sequences=False as it's the last layer

        # Flatten the output of the last LSTM layer
        x = Flatten()(x)

        # Dense layer with ELU activation
        x = Dense(num_classes)(x)  # Fully connected layer
        x = ELU()(x)  # Activation layer (ELU)

        # Softmax for classification (10 classes)
        output = Softmax()(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.01),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model

    def summary(self):
        """
        Prints the summary of the model architecture.
        """
        return self.model.summary()

    def train(self, train_loader, valid_loader, batch_size=1024, epochs=100, callbacks=None, checkpoint_path="best_model.keras"):
        """Trains the model using Keras on data from PyTorch DataLoaders."""
        # Convert train and validation DataLoaders to Numpy
        x_train, y_train = self.dataloader_to_numpy(train_loader)
        x_val, y_val = self.dataloader_to_numpy(valid_loader)
        
        # Convert y_train and y_val to one-hot encoding for Keras if not already done
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_val = to_categorical(y_val, num_classes=self.num_classes)
        
        checkpoint = ModelCheckpoint(checkpoint_path, 
                                     monitor='val_loss', 
                                     save_best_only=True, 
                                     mode='min', 
                                     verbose=1)
        
        # Add checkpoint callback to the list of callbacks
        if callbacks is None:
            callbacks = [checkpoint]
        else:
            callbacks.append(checkpoint)
        
        # Train the Keras model
        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_val, y_val),
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
    
    def dataloader_to_numpy(self, dataloader):
        """Converts a PyTorch DataLoader to Numpy arrays."""
        data_list = []
        target_list = []
        
        for data, target in dataloader:
            data_list.append(data.numpy())
            target_list.append(target.numpy())
        
        data_np = np.concatenate(data_list, axis=0)
        target_np = np.concatenate(target_list, axis=0)
        
        return data_np, target_np