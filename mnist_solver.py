import tensorflow as tf
import tensorflow_datasets as tfds

# Training Params
batch_size = 32 # How many images to send to model at once

# Load the datasets (training and testing)
(ds_train, ds_test), info = tfds.load(name='mnist', split=['train', 'test'], with_info=True)

# Get the number of classes in dataset
num_classes = info.features['label'].num_classes

# Prep the dataset
def sample_prepper(x):
    img = tf.cast(x['image'], tf.float32)/255.0
    img_flat = tf.reshape(img, (-1,))
    label = tf.cast(tf.one_hot(x['label'], num_classes), tf.float32)
    return (img_flat, label)
ds_train = ds_train.map(sample_prepper).batch(batch_size)
ds_test = ds_test.map(sample_prepper).batch(batch_size)

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(28*28,)))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# Builds the model
loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train the model
model.fit(ds_train, epochs=30, validation_data=ds_test)


