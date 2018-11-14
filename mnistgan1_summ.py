from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

#answers = np.load('gan-checks-tf.npz')

#mnist = input_data.read_data_sets('./../cnn2017/GANs/cs231n/datasets/MNIST_data', one_hot=False)
mnist = input_data.read_data_sets('./GANs/cs231n/datasets/MNIST_data', one_hot=False)
def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise
    shape=np.array([batch_size,dim]) 
    return tf.random_normal(shape)

def discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    
    with tf.name_scope("discriminator_data"):
        with tf.variable_scope("discriminator"):
        # TODO: implement architecture
            input_layer = tf.reshape(x, [-1, 28, 28, 1])

            conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],activation=leaky_relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],activation=leaky_relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
            [N, H, Wi, C] = pool2.get_shape().as_list()
        
            pool2_flat = tf.reshape(pool2, [-1, H*Wi*C])#pool2 is of shape(7,7,64)
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=leaky_relu)
            logits = tf.layers.dense(inputs=dense, units=1)
    return logits

def generator(z):
    
    with tf.name_scope("generator_data"):
        with tf.variable_scope("generator"):
            full_1=tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu)
            batch_norm_1=tf.layers.batch_normalization(full_1)
            full_2=tf.layers.dense(inputs=batch_norm_1,units=7*7*128,activation=tf.nn.relu)
            batch_norm_2=tf.layers.batch_normalization(full_2)
            input_layer = tf.reshape(batch_norm_2, [-1, 7, 7, 128])
            [N, H, Wi, C] = input_layer.get_shape().as_list()

            deconv1 = tf.layers.conv2d_transpose(input_layer, filters=64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu)
            bnorm = tf.layers.batch_normalization(deconv1, axis=3)
            img = tf.layers.conv2d_transpose(bnorm, filters=1, kernel_size=4, strides=2, padding="same", activation=tf.nn.tanh)
            img1=tf.reshape(img,[-1,784])
            # img1=tf.reshape(img,[-1,28,28,1])
            # img2=tf.reshape(img,[-1,28,28,1])
    tf.summary.image("generator_data", img, max_outputs=8)
    # tf.summary.image("generator_data", img1, max_outputs=1)    
    return img

def cross_entropy_loss(logits_real, logits_fake):
    with tf.name_scope("cross_entropy"): 
        with tf.name_scope("G_loss"):
            G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
        # tf.summary.scalar("G_loss", G_loss)
        with tf.name_scope("D_loss"):
            D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
            D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
        # tf.summary.scalar("D_loss", D_loss)
    return D_loss, G_loss

def get_solvers(learning_rate=1e-3, beta1=0.5, beta2=0.999):
    with tf.name_scope("Train"):
        D_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1, beta2=beta2)
        G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=beta1, beta2=beta2)
    return D_solver, G_solver


tf.reset_default_graph()

# number of images for each batch
batch_size = 128
# our noise dimension
noise_dim = 96

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, 784])
    z = sample_noise(batch_size, noise_dim)

G_sample = generator(z)

with tf.name_scope("real_fake"):
    with tf.variable_scope("") as scope:
        with tf.name_scope("real_probs"):
            logits_real = discriminator(preprocess_img(x))
        scope.reuse_variables()
        with tf.name_scope("fake_probs"):
            logits_fake = discriminator(G_sample)

with tf.name_scope("variables"):
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

with tf.name_scope("Calculating_loss"):
    D_optimizer, G_optimizer = get_solvers()
    D_loss, G_loss = cross_entropy_loss(logits_real, logits_fake)

D_train_step = D_optimizer.minimize(D_loss, var_list=D_vars)
G_train_step = G_optimizer.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

with tf.name_scope("visualize_original"):
    minibatchs,minbatch_y = mnist.train.next_batch(10)
    new_batch = tf.reshape(minibatchs, [-1,28,28,1])
tf.summary.image("visualize_original",new_batch, max_outputs=1)


# a giant helper function
def run_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, batch_size=128, dim=96,\
            show_every=500, print_every=100, num_epochs=150):

    max_iter = int(mnist.train.num_examples*num_epochs/batch_size)
    with tf.name_scope("running_GAN"):
        for it in range(max_iter):
            with tf.name_scope("Get_batch"):
                minibatch,minbatch_y = mnist.train.next_batch(batch_size)
            tf.summary.image("Get_batch",minibatch[0], max_outputs=1)
            if it % show_every == 99:
                runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, D, G, D_loss_curr, G_loss_curr = sess.run([merged, D_train_step, G_train_step, D_loss, G_loss], options=runoptions, run_metadata=run_metadata, feed_dict={x: minibatch})
                train_writer.add_run_metadata(run_metadata, "step%05d" % it)
                train_writer.add_summary(summary, it)   
                print("Adding run metadata for", it)
            else:
                summary, D, G, D_loss_curr, G_loss_curr = sess.run([merged, D_train_step, G_train_step, D_loss, G_loss], feed_dict={x: minibatch})
                train_writer.add_summary(summary, it)   
                
            if it % print_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    
        train_writer.close()        
    return 

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( "./mnist1_summ13" , sess.graph)
    sess.run(tf.global_variables_initializer())
    run_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step)