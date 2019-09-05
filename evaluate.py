from .capsnet import x_test, y_test,margin_loss,accuracy
import tensorflow as tf
import argparse
import os
import numpy as np

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description="Evaluate Capsule Network on MNIST")
	parser.add_argument("-b", "--batch", default=50, type=int,
						help="Batch size of the training process.")
	parser.add_argument("-d", "--directory", default="/saved_model",
						help="The director the model saving to.")
	parser.add_argument("-n","--name",default="saved_model",
						help="The name of your saved model")
	args = parser.parse_args()
	
	batch_size = args.batch
	n_iterations_test = x_test.shape[0] // batch_size
	
	saver = tf.train.Saver()
	checkpoint_path = os.path.join(args.directory, args.name)

	loss=margin_loss
	X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
	y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

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