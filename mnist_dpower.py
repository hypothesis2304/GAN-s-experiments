from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import scipy.misc

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

def leaky_relu(x, alpha=0.2):
    
    return tf.maximum(x,0.0)+alpha*tf.minimum(x,0.0)

def sample_noise(batch_size, dim):
    
    shape=np.array([batch_size,dim]) 
    # return tf.random_uniform(shape,minval=-1,maxval=1)
    return tf.random_normal(shape)

def gan_loss(logits_real, logits_fake):
    
    D_loss = None
    G_loss = None

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
    
    return D_loss, G_loss
    
def get_solvers(learning_rate=2e-4, beta1=0.5,beta2=0.999):
    
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1,beta2=beta2)
    
    return D_solver, G_solver

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=500, print_every=50, store_every=5000, batch_size=128, num_epoch=0.5):
    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    times=1
    for it in range(max_iter):
        minibatch,minbatch_y = mnist.train.next_batch(batch_size)
        
        if it % store_every == 4999:
            
            runoptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # for times in range(2):                
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            # times = 0
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            summary = sess.run(merged, options=runoptions, run_metadata=run_metadata)
            
            # saver.save(sess, "checkpoint_" + str(times*5) + "000_mnist")
            # times += 1

            train_writer.add_run_metadata(run_metadata, "step%05d" % it)
            train_writer.add_summary(summary, it)   
            print("Adding run metadata for", it)
        
        elif it % show_every == 0:
            # for times in range(2):                
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            # times = 0
            _, G_loss_curr = sess.run([G_train_step, G_loss])
            
            summary = sess.run(merged)
            train_writer.add_summary(summary, it)
        
        else:
            # for times in range(2):                
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            # times = 0
            _, G_loss_curr = sess.run([G_train_step, G_loss])

        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    
    print("Final Images: ")
    print()
    samples = sess.run(G_sample)
    sample_images = samples.reshape((-1, 28,28))
    print(sample_images.shape)
    for i in range(10):
        scipy.misc.imshow(sample_images[i])
    return

def discriminator(x):
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        input_layer = tf.reshape(x, [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[5, 5],strides=[2,2], padding="SAME")
        bnorm1 = tf.layers.batch_normalization(conv1)
        act1 = leaky_relu(bnorm1)   

        conv2 = tf.layers.conv2d(inputs=act1,filters=128, kernel_size=[5, 5], strides=[2,2], padding="SAME")
        bnorm2 = tf.layers.batch_normalization(conv2)
        act2 = leaky_relu(bnorm2)   

        flatten = tf.contrib.layers.flatten(act2)
        
        logits = tf.layers.dense(inputs=flatten, units=1)
        #print(logits.get_shape())
        return logits

def generator(z):
    with tf.name_scope("gen_block"):
        with tf.variable_scope("generator"):
           # TODO: implement architecture

            full_2=tf.layers.dense(inputs=z,units=7*7*128)
            reshaped = tf.reshape(full_2, [-1, 7, 7, 128])
            batch_norm_2=tf.layers.batch_normalization(reshaped)
            act2 = tf.nn.relu(batch_norm_2)
        
            deconv1 = tf.layers.conv2d_transpose(act2, filters=64, kernel_size=4, strides=2, padding="same")
            bnorm = tf.layers.batch_normalization(deconv1, axis=3)
            act3 = tf.nn.relu(bnorm)

            deconv2 = tf.layers.conv2d_transpose(act3, filters=1, kernel_size=4, strides=2, padding="same")
            deconv_final = tf.nn.tanh(deconv2)

            img=tf.reshape(deconv_final,[-1,784])
            
            tf.summary.image("gen_block", deconv_final, max_outputs=6)
        
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
    train_writer = tf.summary.FileWriter( "./scipytest_mnist_summ", sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)
    

