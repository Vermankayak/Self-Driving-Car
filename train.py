import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model

LOGDIR = './save' #save folder stores the final model files at the end of training
#The models traied by AAIC course are in save copy folder and we can use them instead of training a new model.

sess = tf.InteractiveSession()

L2NormConst = 0.001 #It is lambda of L2 regularization

train_vars = tf.trainable_variables()
#loss function used below is =  mean of (y_i - y'_i)^2 + L2 regularization (or L2 norm of weights * lambda)
#We are adding L2 Regularization in order to avoid overfitting
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
#Here we are using Adam optimizer to train our model which will minimize our loss and out learning rate=1e-4 (default value is 1e-3)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op =  tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

# op to write logs to Tensorboard
logs_path = './logs' #Here we are writing all the logs and summary from above into logs file
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 30
batch_size = 100

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size) #We had defined this function in driving_data.py file
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8}) #keep_probability=1-dropout
    if i % 10 == 0: #This is meant to print test loss value or evaluation loss value for every 10th iterations in each epoch.
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt") #we store our model in model.ckpt
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
#Training takes 8.5 hours on non gpu computer and 1.5 hours on GPU system