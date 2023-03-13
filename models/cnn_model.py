import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def cnn_model(classes, input_shape, learning_rate=0.00001):
    tf.keras.backend.clear_session()
    model = models.Sequential()
    model.add(layers.Input((input_shape[0], input_shape[1], 1)))
    model.add(layers.Conv2D(16, 3, activation='relu'))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()
    # metrics の定義
    accuracy = CategoricalAccuracy()
    recall = Recall(class_id=0)
    precision = Precision(class_id=0)

    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=[accuracy, recall, precision, AUC(curve='ROC')],)

    return model