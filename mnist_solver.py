import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
(ds_train, ds_test), info = tfds.load(name='mnist', split=['train', 'test'], with_info=True)
print(info)
# Prep the dataset
def map(x):
    img = tf.cast(x['image'], tf.float32)/255.0
    img_flat = tf.reshape(img, (-1,))
    label = tf.cast(tf.one_hot(x['label'], 10), tf.float32)
    return (img_flat, label)
ds_train = ds_train.map(map).batch(32)
ds_test = ds_test.map(map).batch(32)

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(28*28,)))
# Afterwards, we do automatic shape inference:
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# Builds the model
loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.fit(ds_train, epochs=10, validation_data=ds_test)


