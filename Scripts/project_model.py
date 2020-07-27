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

predictions = model.predict(test_inputs)
print("predictions shape:", predictions.shape)
plt.subplot(2,1,1)
plt.scatter(test_targets[:,6]*100,predictions[:,6]*100,color="blue")
plt.plot([0,100],[0,100],color="red")
plt.xlabel("Measured")
plt.ylabel("Predicted")

plt.subplot(2,1,2)
plt.scatter(test_targets[:,6]*100,(test_targets[:,6]-predictions[:,6])*100,color="blue")
plt.plot([0,100],[0,0],color="red")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig("../Results/image.png")
model.save("../Results/model.h5")
