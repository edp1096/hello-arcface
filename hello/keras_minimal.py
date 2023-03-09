import tensorflow as tf
from tensorflow import keras


device = "gpu" if tf.test.is_gpu_available() else "cpu"

twidth, theight, tchan = 28, 28, 3
im_shape = (1, twidth, theight, tchan)

num_classes = 10

# with tf.device(device):
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=im_shape[1:]),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(num_classes),
    ]
)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(learning_rate=0.05)

model.compile(loss=loss_fn, optimizer=optimizer)
model.summary()


sample_num = 20
inputs = tf.random.uniform((sample_num,) + im_shape[1:], dtype=tf.float32)
labels = tf.random.uniform((sample_num,), minval=0, maxval=num_classes, dtype=tf.int32)


epochs_num = 20
for i in range(epochs_num):
    loss = model.train_on_batch(inputs, labels)
    print(f"epoch: {i+1:>3}, loss: {loss:>8.3f}")


logits = model(inputs)
preds = tf.argmax(logits, axis=1)
print("actual:  pred:")
for actual, pred in zip(labels.numpy(), preds.numpy()):
    print(f"   {actual:>2}       {pred:>2}")
