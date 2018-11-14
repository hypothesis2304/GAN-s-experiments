from __future__ import print_function, division
import tensorflow as tf
import numpy as np


def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./GANs/cs231n/datasets/MNIST_data', one_hot=False)

def leaky_relu(x, alpha=0.02):
    
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    
    shape=np.array([batch_size,dim]) 
    #return tf.random_uniform(shape,minval=-1,maxval=1)
    return tf.random_normal(shape)

def gan_loss(logits_real, logits_fake):
    
    D_loss = None
    G_loss = None
    g=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.ones_like(logits_fake))
    d1=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,labels=tf.ones_like(logits_real))
    d2=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,labels=tf.zeros_like(logits_fake))
    d=d1+d2
    D_loss=tf.reduce_mean(d)
    G_loss=tf.reduce_mean(g)
    return D_loss, G_loss

def get_solvers(learning_rate=1e-3, beta1=0.5,beta2=0.999):
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=2*learning_rate,beta1=beta1,beta2=beta2)
    
    return D_solver, G_solver

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=500, print_every=50, batch_size=128, num_epoch=30):
    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    
    for it in range(max_iter):
        minibatch,minbatch_y = mnist.train.next_batch(batch_size)
        if it % show_every == 99:
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            summary = sess.run(merged, options=runoptions, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step%05d" % it)
            train_writer.add_summary(summary, it)   
            print("Adding run metadata for", it)
        else:
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            _, G_loss_curr = sess.run([G_train_step, G_loss])

        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return

def discriminator(x):
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
        #print(logits.get_shape())
        return logits

def generator(z):
    with tf.name_scope("gen_block"):
        with tf.variable_scope("generator"):
           # TODO: implement architecture
            full_1=tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu)
            batch_norm_1=tf.layers.batch_normalization(full_1)
        #print(batch_norm_1.get_shape())
            full_2=tf.layers.dense(inputs=batch_norm_1,units=7*7*128,activation=tf.nn.relu)
            batch_norm_2=tf.layers.batch_normalization(full_2)
        
        #input_layer = tf.reshape(batch_norm_2, [-1, 28, 28, 8])
            input_layer = tf.reshape(batch_norm_2, [-1, 7, 7, 128])
        
            [N, H, Wi, C] = input_layer.get_shape().as_list()
        
            deconv1 = tf.layers.conv2d_transpose(input_layer, filters=64, kernel_size=4, strides=2, padding="same", activation=tf.nn.relu)
            bnorm = tf.layers.batch_normalization(deconv1, axis=3)
            deconv2 = tf.layers.conv2d_transpose(bnorm, filters=1, kernel_size=4, strides=2, padding="same", activation=tf.nn.tanh)
        # print(img.get_shape())
            img=tf.reshape(deconv2,[-1,784])
            
            tf.summary.image("gen_block", deconv2, max_outputs=6)
        # print(img.get_shape())
        
            return img

tf.reset_default_graph()

batch_size = 128
# our noise dimension
noise_dim = 96

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(preprocess_img(x))
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    print(G_sample.get_shape())
    logits_fake = discriminator(G_sample)
    
# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator') 

D_solver,G_solver = get_solvers()
D_loss, G_loss = gan_loss(logits_real, logits_fake)
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS,'generator')

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( "./mnist_lr_summ4", sess.graph)
    sess.run(tf.global_variables_initializer())
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)