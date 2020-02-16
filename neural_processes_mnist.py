import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import true_neural_processes as NP
import collections
import numpy as np
import random
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))

training_iterations=1000000 #@param {type:"number"}
max_context_points=200
PLOT_AFTER = 1000 #@param {type:"number"}
HIDDEN_SIZE = 128 #@param {type:"number"}
MODEL_TYPE = 'ANP' #@param ['NP','ANP']
ATTENTION_TYPE = 'multihead' #@param ['uniform','laplace','dot_product','multihead']
random_kernel_parameters=True #@param {type:"boolean"}
num_batches=1


target_x=np.asarray([(a, b) for a in range(28) for b in range(28)])/10
target_x=np.expand_dims(target_x,0)
target_x=np.tile(target_x,(num_batches,1,1))



# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
tf.reset_default_graph()


context_x_pl=tf.placeholder(shape=[num_batches,None,2], dtype=tf.float32, name='context_x')
context_y_pl=tf.placeholder(shape=[num_batches,None,1], dtype=tf.float32, name='context_y')
target_x_pl=tf.placeholder(shape=[num_batches,784,2], dtype=tf.float32, name='target_x')
target_y_pl=tf.placeholder(shape=[num_batches,784,1], dtype=tf.float32, name='target_y')

query = ((context_x_pl, context_y_pl), target_x_pl)

data_train=NPRegressionDescription(
            query=query,
            target_y=target_y_pl,
            num_total_points=tf.shape(target_x_pl)[1],
            num_context_points=tf.shape(context_x_pl)[1])

context_x_pl_test=tf.placeholder(shape=[num_batches,None,2], dtype=tf.float32, name='context_x_test')
context_y_pl_test=tf.placeholder(shape=[num_batches,None,1], dtype=tf.float32, name='context_y_test')
target_x_pl_test=tf.placeholder(shape=[num_batches,784,2], dtype=tf.float32, name='target_x_test')
target_y_pl_test=tf.placeholder(shape=[num_batches,784,1], dtype=tf.float32, name='target_y_test')

query_test = ((context_x_pl_test, context_y_pl_test), target_x_pl_test)

data_test=NPRegressionDescription(
            query=query,
            target_y=target_y_pl_test,
            num_total_points=tf.shape(target_x_pl_test)[1],
            num_context_points=tf.shape(context_x_pl_test)[1])



latent_encoder_output_sizes = [HIDDEN_SIZE]*4
num_latents = HIDDEN_SIZE
deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]
use_deterministic_path = True

# ANP with multihead attention
if MODEL_TYPE == 'ANP':
  attention = NP.Attention(rep='mlp', output_sizes=[HIDDEN_SIZE]*2,
                        att_type=ATTENTION_TYPE)
# NP - equivalent to uniform attention
elif MODEL_TYPE == 'NP':
  attention = NP.Attention(rep='identity', output_sizes=None, att_type='uniform')
else:
  raise NameError("MODEL_TYPE not among ['ANP,'NP']")

# Define the model
model = NP.LatentModel(latent_encoder_output_sizes, num_latents,
                    decoder_output_sizes, use_deterministic_path,
                    deterministic_encoder_output_sizes, attention)

# Define the loss
_, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                 data_train.target_y)

# Get the predicted mean and variance at the target points for the testing set
mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)

# Set up the optimizer and train step
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()

# Train and plot
with tf.train.MonitoredSession() as sess:
  sess.run(init)

  for i in range(training_iterations):
      print(i)
      target_y, _ = mnist.train.next_batch(num_batches)
      target_y=np.expand_dims(target_y,-1)
      num_context=random.choice(list(range(max_context_points)))+1
      context=random.sample(list(range(784)),num_context)
      context_x=target_x[:,context,:]
      context_y=target_y[:,context,:]

      if i % PLOT_AFTER != 0:
          sess.run([train_step], feed_dict={target_x_pl: target_x, target_y_pl: target_y, context_x_pl: context_x,
                                            context_y_pl: context_y})

          # Plot the predictions in `PLOT_AFTER` intervals
      else:

          loss_value, pred_y, std_y, target_y, whole_query = sess.run(
              [loss, mu, sigma, data_test.target_y,
               data_test.query],
              feed_dict={target_x_pl_test: target_x, target_y_pl_test: target_y, context_x_pl_test: context_x,
                         context_y_pl_test: context_y, target_x_pl: target_x, target_y_pl: target_y,
                         context_x_pl: context_x, context_y_pl: context_y})

          print('Iteration: {}, loss: {}'.format(i, loss_value))

          plt.imshow(np.reshape(pred_y[0, :, :], (28, 28)),cmap='Greys')
          plt.show()









