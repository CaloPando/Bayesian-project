import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import true_neural_processes as NP

'''
This is a simulation of a classification problem where we generate a set of random points on a plane and set
those that fall inside a circle to 1, to 0 otherwise, to divide them in two categories.
The radius of the circle is random, and the NPs learn to adapt to it
'''

TRAINING_ITERATIONS = 100000 #@param {type:"number"}
MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
PLOT_AFTER = 100 #@param {type:"number"}
HIDDEN_SIZE = 128 #@param {type:"number"}
MODEL_TYPE = 'ANP' #@param ['NP','ANP']
ATTENTION_TYPE = 'laplace' #@param ['uniform','laplace','dot_product','multihead']
random_kernel_parameters=True #@param {type:"boolean"}


tf.reset_default_graph()


dataset_train = NP.classification(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS)
data_train = dataset_train.Create_dataset(L=2)

dataset_test = NP.classification(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
data_test = dataset_test.Create_dataset(L=2)



# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
latent_encoder_output_sizes = [HIDDEN_SIZE]*4
num_latents = 32
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

  for it in range(TRAINING_ITERATIONS):
    sess.run([train_step])

    # Plot the predictions in `PLOT_AFTER` intervals
    if it % PLOT_AFTER == 0:
      loss_value, pred_y, std_y, target_y, whole_query = sess.run(
          [loss, mu, sigma, data_test.target_y,
           data_test.query])

      (context_x, context_y), target_x = whole_query
      print('Iteration: {}, loss: {}'.format(it, loss_value))

      radius=sess.run(data_test.radius)
      print(pred_y)


      def f(x):
          return np.sqrt(radius ** 2 - x ** 2)

      x = np.linspace(-radius, radius, 100)
      y_up = f(x)
      y_down = -f(x)

      positives = [index for index, value in enumerate(pred_y[0,:]) if value > 0.5]
      negatives = [index for index, value in enumerate(pred_y[0,:]) if value < 0.5]

      plt.plot(x, y_up, 'k--')
      plt.plot(x, y_down, 'k--')
      plt.fill_between(x, y_down, y_up, color='#539ecd')
      plt.scatter(context_x[0, :, 0], context_x[0, :, 1], marker='s', c='g')
      positives_plot=plt.scatter(target_x[0,positives, 0], target_x[0,positives, 1], marker='x', c='b')
      negatives_plot=plt.scatter(target_x[0,negatives, 0], target_x[0,negatives, 1], marker='o', c='r')
      context_plot=plt.scatter(context_x[0, :, 0], context_x[0, :, 1], marker='s', c='k')
      plt.legend([positives_plot, negatives_plot, context_plot], ["1", "0", "context"],loc="upper left")
      plt.xlim([-1, 1])
      plt.ylim([-1, 1])
      plt.show()










