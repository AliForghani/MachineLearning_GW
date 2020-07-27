import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_seed(500)

npz = np.load('../Results/data_train.npz')

train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.float)

npz = np.load('../Results/data_validation.npz')

validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.float)

npz = np.load('../Results/data_test.npz')

test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.float)


input_size = 6
output_size = 8

hidden_layer_size =200

model = tf.keras.Sequential([

    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(output_size, activation='linear')])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

batch_size = 100

max_epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=1)

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size,
          epochs=max_epochs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose = 2 )
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

model.save("../Results/model.h5")
