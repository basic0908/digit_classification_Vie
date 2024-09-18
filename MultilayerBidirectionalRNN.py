from keras.layers import Input, Bidirectional, LSTM, Dense, Flatten, ELU, Softmax, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
from keras.metrics import Precision, Recall



class MultilayerBidirectionalLSTM:
    def __init__(self, input_shape=(14, 256), num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        print("this is 1 dropout version")
        """
        Build the LSTM model based on the EPOC architecture.
        :param input_shape: Shape of the input data (timesteps, features)
        :param num_classes: Number of output classes
        :return: Keras model
        """
        model = Sequential()

        # Bidirectional LSTM Layer 1
        model.add(Bidirectional(LSTM(256, dropout=0.1, return_sequences=True), input_shape=(14, 256)))

        # Bidirectional LSTM L1ayer 2
        model.add(Bidirectional(LSTM(128, dropout=0.1, return_sequences=True)))


        # Bidirectional LSTM Layer 3
        model.add(Bidirectional(LSTM(64, dropout=0.1, return_sequences=False)))


        # Flatten the output for Dense layer
        model.add(Flatten())

        # Dense layer for classification with softmax activation
        model.add(Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', Precision(), Recall()])
        
        return model

    def summary(self):
        """
        Prints the summary of the model architecture.
        """
        return self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, batch_size=1024, epochs=100, callbacks=None, checkpoint_path="best_model.keras"):
        """Trains the model using Keras on Numpy arrays."""
        print("val_accuracy version")
        checkpoint = ModelCheckpoint(checkpoint_path, 
                                    monitor='val_accuracy', 
                                    save_best_only=True, 
                                    mode='max', 
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

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)

        # Evaluate using a metric, e.g., accuracy score
        from sklearn.metrics import accuracy_score

        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy}")
            
    def predict(self, x_input):
        """Predicts the class labels for the given input data."""
        
        # Use the trained model to predict the class probabilities
        predictions = self.model.predict(x_input)
        
        # Get the predicted class labels (by choosing the class with the highest probability)
        predicted_labels = predictions.argmax(axis=1)
        
        return predicted_labels, predictions