import tensorflow as tf
from capsnet import CapsNet, reconstruct, safe_norm
# import capsnet.CapsNet
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

Labels = 10
outputDimension = 16

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

parser = argparse.ArgumentParser(description="Predict Capsule Network on MNIST")
parser.add_argument("-d", "--directory", default="saved_model",
					help="Directory the model saved.")
parser.add_argument("-n", "--name", default="saved_model",
					help="The name of your model.")
parser.add_argument("-r", "--reconstruct", default=True,
					help="Reconstruct or not.")
args = parser.parse_args()

checkpoint_path = os.path.join(args.directory, args.name)
n_samples = 5

idx = np.random.choice(x_test.shape[0], size=n_samples, replace=False)
sample_images = x_test[idx, :]
sample_images = sample_images.reshape(-1, 28, 28,1)

# Placeholders
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

# Rebuild the models.
caps2_output = CapsNet(X)
y_prob = safe_norm(caps2_output, axis=-2, name="y_prob")
# Choose the predicted one.
y_prob_argmax = tf.argmax(y_prob, axis=2, name="y_predicted_argmax")
y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2], name="y_pred")

if args.reconstruct:
	reconstruction_loss, decoder_output = reconstruct(caps2_output, mask_with_labels, X, y, y_pred, Labels, outputDimension)

saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, checkpoint_path)
	if args.reconstruct:
		caps2_output_value, decoder_output_value, y_pred_value = sess.run(
			[caps2_output, decoder_output, y_pred],
			feed_dict={X: sample_images,
					   y: np.array([], dtype=np.int64)})
	else:
		caps2_output_value, y_pred_value = sess.run(
			[caps2_output, y_pred],
			feed_dict={X: sample_images,
					   y: np.array([], dtype=np.int64)})

	sample_images = sample_images.reshape(-1, 28, 28)
	if args.reconstruct:
		reconstructions = decoder_output_value.reshape([-1, 28, 28])

	plt.figure(figsize=(n_samples * 2, 3))
	for index in range(n_samples):
		plt.subplot(1, n_samples, index + 1)
		plt.imshow(sample_images[index], cmap="binary")
		plt.title("Label:" + str(y_test[idx, index]))
		plt.axis("off")

	plt.savefig(args.directory + "/initial.png")

	plt.figure(figsize=(n_samples * 2, 3))
	for index in range(n_samples):
		plt.subplot(1, n_samples, index + 1)
		plt.title("Predicted:" + str(y_pred_value[index]))
		if args.reconstruct:
			plt.imshow(reconstructions[index], cmap="binary")
		plt.axis("off")
	plt.savefig(checkpoint_path + "/predict.png")
