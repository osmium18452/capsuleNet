import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set random seeds to ensure each running presents the same result.
np.random.seed(42)
tf.set_random_seed(42)


# Safe squash function.
def squash(s, axis=-1, epsilon=1e-7, name=None):
	with tf.name_scope(name, default_name="squash"):
		squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
		safe_norm = tf.sqrt(squared_norm + epsilon)
		squash_factor = squared_norm / (1. + squared_norm)
		unit_vector = s / safe_norm
		return squash_factor * unit_vector


def CapsNet(X):
	# First layer, convolutional.
	conv1_params = {
		"filters": 256,
		"kernel_size": 9,
		"strides": 1,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv1"
	}
	# Use 256 9*9 filters to extract features in the first conv layer.
	conv1 = tf.layers.conv2d(X, **conv1_params)

	# Primary layer. Contains a convolution layer and the first cpasule layer.\
	# We extract 32 features and each feature will be casted to 6*6 capsules, whose dimension is [8].
	caps1_maps = 32
	caps1_caps = caps1_maps * 6 * 6
	caps1_dims = 8
	conv2_params = {
		"filters": caps1_maps * caps1_dims,
		"kernel_size": 9,
		"strides": 2,
		"padding": "valid",
		"activation": tf.nn.relu,
		"name": "conv2"
	}
	conv2 = tf.layers.conv2d(conv1, **conv2_params)
	caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name="caps1_raw")
	# Squash the capsules.
	caps1_output = squash(caps1_raw, name="caps1_output")

	# Digit layer
	# 10 capsule in the digit layer and each capsule outputs a vector with dimension of 16.
	caps2_caps = 10
	caps2_dims = 16
	init_sigma = 0.1
	w_init = tf.random_normal(
		shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
		stddev=init_sigma, dtype=tf.float32, name="W_init"
	)
	W = tf.Variable(w_init, name="W")
	# Tile the W to fit to the batch.
	batch_size = tf.shape(X)[0]
	w_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
	# To use the tf.matmul() function, we have to make the Primary layer's dimension fit to the W_tile tensor.
	caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded1")
	caps1_output_expanded = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_expanded2")
	caps1_output_tiled = tf.tile(caps1_output_expanded, [1, 1, caps2_caps, 1, 1], name="caps1_output_tiled")
	caps2_predicted = tf.matmul(w_tiled, caps1_output_tiled, name="caps2_predicted")

	# Dynamic routing.
	# FIXME: There may be something wrong...
	# Initialize b_i,j
	raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=np.float32, name="raw_weights")

	# First round.
	# Use softmax to calculate c=softmax(b).
	routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
	# Multiply and sum
	weighted_prediction = tf.multiply(routing_weights, caps2_predicted, name="weighted_prediction")
	weighted_sum = tf.reduce_sum(weighted_prediction, axis=1, keep_dims=True, name="weighted_sum")
	caps2_output_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

	# Second round
	caps2_output_1_tiled = tf.tile(caps2_output_1, [1, caps1_caps, 1, 1, 1], name="caps2_1st_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_1_tiled, transpose_a=True, name="agreement")
	raw_weights_round2 = tf.add(raw_weights, agreement, name="raw_weight_round_2")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round2 = tf.nn.softmax(raw_weights_round2, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round2 = tf.multiply(routing_weights_round2, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round2 = tf.reduce_sum(weighted_prediction_round2, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_2 = squash(weighted_sum_round2, axis=-2, name="caps2_output_round_2")

	# Third round
	caps2_output_3_tiled = tf.tile(caps2_output_2, [1, caps1_caps, 1, 1, 1], name="caps2_2nd_output_tiled")
	# Update the new b_i,j
	agreement = tf.matmul(caps2_predicted, caps2_output_3_tiled, transpose_a=True, name="agreement")
	raw_weights_round3 = tf.add(raw_weights_round2, agreement, name="raw_weight_round_3")
	# Use softmax to calculate c=softmax(b).
	routing_weights_round3 = tf.nn.softmax(raw_weights_round3, dim=2, name="routing_weights_round_2")
	# Multiply and sum
	weighted_prediction_round3 = tf.multiply(routing_weights_round3, caps2_predicted,
											 name="weighted_prediction_round_2")
	weighted_sum_round3 = tf.reduce_sum(weighted_prediction_round3, axis=1, keep_dims=True, name="weighted_sum_round_2")
	caps2_output_3 = squash(weighted_sum_round3, axis=-2, name="caps2_output_round_3")

	return caps2_output_3


def safe_norm(s, axis=-1, epslion=1e-7, keep_dims=False, name=None):
	with tf.name_scope(name, default_name="safenorm"):
		squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
		return tf.sqrt(squared_norm + epslion)


def reconstruct(capsOutput, mask_with_labels, X, y, y_pred, Labels, outputDimension):
	# A normal 3-layer fully connected neuron network.
	# Mask
	# mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
	reconstruction_target = tf.cond(mask_with_labels, lambda: y, lambda: y_pred, name="reconstruction_target")
	reconstruction_mask = tf.one_hot(reconstruction_target, depth=Labels, name="reconstruction_mask")
	reconstruction_mask_reshaped = tf.reshape(
		reconstruction_mask, [-1, 1, Labels, 1, 1],
		name="reconstruction_mask_reshaped")
	capsOutput_masked = tf.multiply(
		capsOutput, reconstruction_mask_reshaped,
		name="caps2_output_masked")
	decoder_input = tf.reshape(capsOutput_masked, [-1, Labels * outputDimension])

	# Decoder
	# Tow relu and a sigmoid
	n_hidden1 = 512
	n_hidden2 = 1024
	n_output = 28 * 28

	with tf.name_scope("decoder"):
		hidden1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name="hidden1")
		hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
		decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name="decoder_output")

	# Reconstruction loss.
	X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
	squared_difference = tf.square(X_flat - decoder_output, name="squared_difference")
	reconstruction_loss = tf.reduce_mean(squared_difference, name="reconstruction_loss")

	return reconstruction_loss, decoder_output


if __name__ == "__main__":

	import os
	import argparse

	parser = argparse.ArgumentParser(description="Capsule Network on MNIST")
	parser.add_argument("-e", "--epochs", default=10, type=int,
						help="The number of epochs you want to train.")
	parser.add_argument("-b", "--batch", default=50, type=int,
						help="Batch size of the training process.")
	parser.add_argument("-g", "--gpu", default="0", type=str,
						help="Which gpu(s) you want to use.")
	parser.add_argument("--showimg", default=False, type=bool,
						help="Whether to show the first n training images.")
	parser.add_argument("--nimg", default=5, type=int,
						help="How many images you want to show.")
	parser.add_argument("--restore", default=False,type=bool,
						help="Restore the trained model or not.")
	parser.add_argument("-d", "--directory", default="./saved_model",
						help="The director the model saving to.")
	parser.add_argument("-r", "--reconstruct", default=False,type=bool,
						help="Reconstruct the image.")
	parser.add_argument("-n", "--name", default="saved_model",
						help="The name of your model.")
	parser.add_argument("-o", "--only_eval", default=False,
						help="Only evaluate the model.")
	args = parser.parse_args()
	print(args)

	# Load dataset.
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	if args.showimg:
		n_samples = args.nimg
		for index in range(n_samples):
			plt.subplot(1, n_samples, index + 1)
			sample_image = x_train[index].reshape(28, 28)
			plt.imshow(sample_image, cmap="binary")
			plt.axis("off")
		plt.show()
		print(y_train[:n_samples])

	# TODO: Modify the properties of X to fit other datasets.
	X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
	y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

	capsOutput = CapsNet(X)

	# Calculate the probability.
	y_prob = safe_norm(capsOutput, axis=-2, name="y_prob")
	# Choose the predicted one.
	y_prob_argmax = tf.argmax(y_prob, axis=2, name="y_predicted_argmax")
	y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2], name="y_pred")

	Labels = 10
	outputDimension = 16

	# TODO: Modulize the loss function and the reconstruction process.
	# Training preparation
	# Define the loss function. Here we use the margin loss.
	mPlus = 0.9
	mMinus = 0.1
	lambda_ = 0.5

	# TODO: These two pramaters can be changed to fit into new datasets.

	T = tf.one_hot(y, depth=Labels, name="T")
	capsOutput_norm = safe_norm(capsOutput, axis=-2, keep_dims=True, name="capsule_output_norm")

	present_error_raw = tf.square(tf.maximum(0., mPlus - capsOutput_norm), name="present_error_raw")
	present_error = tf.reshape(present_error_raw, shape=(-1, 10), name="present_error")
	absent_error_raw = tf.square(tf.maximum(0., capsOutput_norm - mMinus), name="absent_error_raw")
	absent_error = tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error")

	L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
	margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

	# Reconstruction
	if args.reconstruct:
		mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
		reconstruction_loss, decoder_output = reconstruct(capsOutput, mask_with_labels, X, y, y_pred, Labels, outputDimension)

		# Final loss
		alpha = 0.00001  # The alpha parameter will ensure that the margin loss can dominate the training process.
		loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
	else:
		loss = margin_loss

	# Accuracy
	correct = tf.equal(y, y_pred, name="correct")
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss, name="training_op")

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	# Training...
	n_epochs = args.epochs
	batch_size = args.batch
	restore_checkpoint = args.restore

	if not os.path.exists(args.directory):
		os.makedirs(args.directory)

	n_iterations_per_epoch = x_train.shape[0] // batch_size
	n_iterations_validation = x_test.shape[0] // batch_size
	best_loss_val = np.infty
	checkpoint_path = os.path.join(args.directory, args.name)

	if not args.only_eval:
		with tf.Session() as sess:
			if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
				saver.restore(sess, checkpoint_path)
				print()
				print("model restored")
				print()
			else:
				init.run()

			for epoch in range(n_epochs):
				for iteration in range(1, n_iterations_per_epoch + 1):
					idx = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
					x_batch = x_train[idx, :]
					y_batch = y_train[idx]
					# X_batch, y_batch = mnist.train.next_batch(batch_size)
					# Train and validate the loss
					if args.reconstruct:
						_, loss_train = sess.run(
							[training_op, loss],
							feed_dict={X: x_batch.reshape([-1, 28, 28, 1]),
									   y: y_batch,
									   mask_with_labels: True})
					else:
						_, loss_train = sess.run(
							[training_op, loss],
							feed_dict={X: x_batch.reshape([-1, 28, 28, 1]),
									   y: y_batch})
					print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
						iteration, n_iterations_per_epoch,
						iteration * 100 / n_iterations_per_epoch,
						loss_train),
						end="")

				# Evaluate the model after each epoch.
				loss_vals = []
				acc_vals = []
				for iteration in range(1, n_iterations_validation + 1):
					idx = np.random.choice(x_test.shape[0], size=batch_size, replace=False)
					x_batch_test = x_test[idx, :]
					y_batch_test = y_test[idx]
					loss_val, acc_val = sess.run(
						[loss, accuracy],
						feed_dict={X: x_batch_test.reshape([-1, 28, 28, 1]),
								   y: y_batch_test})
					loss_vals.append(loss_val)
					acc_vals.append(acc_val)
					print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
						iteration, n_iterations_validation,
						iteration * 100 / n_iterations_validation),
						end=" " * 10)
				loss_val = np.mean(loss_vals)
				acc_val = np.mean(acc_vals)
				print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
					epoch + 1, acc_val * 100, loss_val,
					" (improved)" if loss_val < best_loss_val else ""))

				# Save the model is the model is the best at present.
				if loss_val < best_loss_val:
					save_path = saver.save(sess, checkpoint_path)
					best_loss_val = loss_val

	n_iterations_test = x_test.shape[0] // batch_size
	with tf.Session() as sess:
		saver.restore(sess, checkpoint_path)

		loss_tests = []
		acc_tests = []
		for iteration in range(1, n_iterations_test + 1):
			idx = np.random.choice(x_test.shape[0], size=batch_size, replace=False)
			x_batch = x_test[idx, :]
			y_batch = y_test[idx]
			loss_test, acc_test = sess.run(
				[loss, accuracy],
				feed_dict={X: x_batch.reshape([-1, 28, 28, 1]),
						   y: y_batch})
			loss_tests.append(loss_test)
			acc_tests.append(acc_test)
			print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
				iteration, n_iterations_test,
				iteration * 100 / n_iterations_test),
				end=" " * 10)
		loss_test = np.mean(loss_tests)
		acc_test = np.mean(acc_tests)
		print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
			acc_test * 100, loss_test))
