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
    # Cast image from uint8 (0-255) to float32 (0-1)
    img = tf.cast(x['image'], tf.float32)/255.0
    # Reshape the image from (28,28,1) to (784)
    img_flat = tf.reshape(img, (-1,))
    # Change numeric labels to one-hot representations (from 3 to [0,0,0,1])
    label = tf.cast(tf.one_hot(x['label'], num_classes), tf.float32)
    # Return the sample tuple
    return (img_flat, label)

# Setup the dataset pipeline for training and testing 
ds_train = ds_train.map(sample_prepper).batch(batch_size)
ds_test = ds_test.map(sample_prepper).batch(batch_size)

# Define the model
model = tf.keras.Sequential()
# Adding fully-connected (dense) layer with 32 units and input shape (784,)
model.add(tf.keras.layers.Dense(32, input_shape=(28*28,)))
model.add(tf.keras.layers.Dense(32))
# Last layer gives a vector of values that sum to 1, each value corresponding to a class probability
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# Builds the model
loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train the model
model.fit(ds_train, epochs=30, validation_data=ds_test)


