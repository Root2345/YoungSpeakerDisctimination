import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def lstm_model(classes, learning_rate=1e-6):
    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Input((257, 251)))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(32, return_sequences=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()
    accuracy = CategoricalAccuracy()
    recall = Recall()
    precision = Precision()
    auc = AUC()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=[accuracy, recall, precision, auc])

    return model
