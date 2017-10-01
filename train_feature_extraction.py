import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from alexnet import AlexNet

# TODO: Load traffic signs data.
data = None
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.
train_features, valid_features, train_classes, valid_classes = train_test_split(data['features'], data['labels'],
                                                                                test_size=0.33)
nb_classes = len(np.unique(train_classes))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# TODO: Add the final layer for traffic sign classification.
mu = 0
sigma = 0.1
fc8_W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
fc8_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8_W) + fc8_b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001

y = tf.placeholder(tf.int32, (None), name="Y")
one_hot_y = tf.one_hot(y, nb_classes)
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
EPOCHS = 3
BATCH_SIZE = 128


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()
    for epoch in range(EPOCHS):
        train_features, train_classes = shuffle(train_features, train_classes)
        for offset in tqdm(range(0, len(train_features), BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batch_x, batch_y = train_features[offset:end], train_classes[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(valid_features, valid_classes)
        epoch_number = epoch + 1
        print("EPOCH {} ...".format(epoch_number))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
