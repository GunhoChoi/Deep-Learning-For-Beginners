# This code is in Python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

def generator(z):
    
    zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    zCon = tf.reshape(zP,[-1,4,4,256])
    
    gen1 = slim.convolution2d(\
        zCon,num_outputs=128,kernel_size=[3,3],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
    gen1 = tf.depth_to_space(gen1,2)
    
    gen2 = slim.convolution2d(\
        gen1,num_outputs=64,kernel_size=[3,3],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
    gen2 = tf.depth_to_space(gen2,2)
    
    gen3 = slim.convolution2d(\
        gen2,num_outputs=32,kernel_size=[3,3],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)
    gen3 = tf.depth_to_space(gen3,2)
    
    g_out = slim.convolution2d(\
        gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",\
        biases_initializer=None,activation_fn=tf.nn.tanh,\
        scope='g_out', weights_initializer=initializer)
    
    return g_out

def discriminator(bottom, cat_list, conts, reuse=False):
    
    # tf.depth_to_space(input, block_size, name=None) 
    # That is, assuming the input is in the shape: [batch, height, width, depth], the shape of the output will be: 
    # [batch, height*block_size, width*block_size, depth/(block_size*block_size)]

    dis1 = slim.convolution2d(bottom,32,[3,3],padding="SAME",\
        biases_initializer=None,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv1',weights_initializer=initializer)
    dis1 = tf.space_to_depth(dis1,2)
    
    dis2 = slim.convolution2d(dis1,64,[3,3],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv2', weights_initializer=initializer)
    dis2 = tf.space_to_depth(dis2,2)
    
    dis3 = slim.convolution2d(dis2,128,[3,3],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv3',weights_initializer=initializer)
    dis3 = tf.space_to_depth(dis3,2)
        
    dis4 = slim.fully_connected(slim.flatten(dis3),1024,activation_fn=lrelu,\
        reuse=reuse,scope='d_fc1', weights_initializer=initializer)
        
    d_out = slim.fully_connected(dis4,1,activation_fn=tf.nn.sigmoid,\
        reuse=reuse,scope='d_out', weights_initializer=initializer)
    
    q_a = slim.fully_connected(dis4,128,normalizer_fn=slim.batch_norm,\
        reuse=reuse,scope='q_fc1', weights_initializer=initializer)
    
    
    ## Here we define the unique layers used for the q-network. The number of outputs depends on the number of 
    ## latent variables we choose to define.
    q_cat_outs = []
    for idx,var in enumerate(cat_list):
        q_outA = slim.fully_connected(q_a,var,activation_fn=tf.nn.softmax,\
            reuse=reuse,scope='q_out_cat_'+str(idx), weights_initializer=initializer)
        q_cat_outs.append(q_outA)
    
    q_cont_outs = None
    if conts > 0:
        q_cont_outs = slim.fully_connected(q_a,conts,activation_fn=tf.nn.tanh,\
            reuse=reuse,scope='q_out_cont_'+str(conts), weights_initializer=initializer)
    
    return d_out,q_cat_outs,q_cont_outs

tf.reset_default_graph()

z_size = 64 #Size of initial z vector used for generator.

# Define latent variables.
categorical_list = [10] # Each entry in this list defines a categorical variable of a specific size.
number_continuous = 2 # The number of continous variables.

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32) #Real images

#These placeholders load the latent variables.
latent_cat_in = tf.placeholder(shape=[None,len(categorical_list)],dtype=tf.int32)
latent_cat_list = tf.split(1,len(categorical_list),latent_cat_in)
latent_cont_in = tf.placeholder(shape=[None,number_continuous],dtype=tf.float32)

oh_list = []
for idx,var in enumerate(categorical_list):
    latent_oh = tf.one_hot(tf.reshape(latent_cat_list[idx],[-1]),var)
    oh_list.append(latent_oh)


#Concatenate all c and z variables.
z_lats = oh_list[:]
z_lats.append(z_in)
z_lats.append(latent_cont_in)
z_lat = tf.concat(1,z_lats)

Gz = generator(z_lat) #Generates images from random z vectors
Dx,_,_ = discriminator(real_in,categorical_list,number_continuous) #Produces probabilities for real images
Dg,QgCat,QgCont = discriminator(Gz,categorical_list,number_continuous,reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log((Dg/(1-Dg)))) #KL Divergence optimizer

#Combine losses for each of the categorical variables.
cat_losses = []
for idx,latent_var in enumerate(oh_list):
    cat_loss = -tf.reduce_sum(latent_var*tf.log(QgCat[idx]),reduction_indices=1)
    cat_losses.append(cat_loss)
    
#Combine losses for each of the continous variables.
if number_continuous > 0:
    q_cont_loss = tf.reduce_sum(0.5 * tf.square(latent_cont_in - QgCont),reduction_indices=1)
else:
    q_cont_loss = tf.constant(0.0)

q_cont_loss = tf.reduce_mean(q_cont_loss)
q_cat_loss = tf.reduce_mean(cat_losses)
q_loss = tf.add(q_cat_loss,q_cont_loss)

tvars = tf.trainable_variables()

#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.002,beta1=0.5)
trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:-2-((number_continuous>0)*2)-(len(categorical_list)*2)]) #Only update the weights for the discriminator network.
g_grads = trainerG.compute_gradients(g_loss, tvars[0:9]) #Only update the weights for the generator network.
q_grads = trainerG.compute_gradients(q_loss, tvars) 

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)
update_Q = trainerQ.apply_gradients(q_grads)

batch_size = 20 #Size of image batch to apply at each iteration.
iterations = 500000 #Total number of iterations to use.
sample_directory = './figsInfoGAN' #Directory to save sample images from generator in.
model_directory = './modelsInfoGAN' #Directory to save trained model to.


init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    for i in range(1):
        zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
        lcat = np.random.randint(0,10,[batch_size,len(categorical_list)]) #Generate random c batch
        lcont = np.random.uniform(-1,1,[batch_size,number_continuous]) #
        
        xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
        xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
        
        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs,latent_cat_in:lcat,latent_cont_in:lcont}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs,latent_cat_in:lcat,latent_cont_in:lcont}) #Update the generator, twice for good measure.
        _,qLoss,qK,qC = sess.run([update_Q,q_loss,q_cont_loss,q_cat_loss],feed_dict={z_in:zs,latent_cat_in:lcat,latent_cont_in:lcont}) #Update to optimize mutual information.
        
        a=sess.run(z_lat,feed_dict={z_in:zs,real_in:xs,latent_cat_in:lcat,latent_cont_in:lcont})
        print(a.shape)

        if i % 100 == 0:
            print ("Gen Loss: " + str(gLoss) + " Dis Loss: " + str(dLoss) + " Q Losses: " + str([qK,qC]))
            z_sample = np.random.uniform(-1.0,1.0,size=[100,z_size]).astype(np.float32) #Generate another z batch
            lcat_sample = np.reshape(np.array([e for e in range(10) for _ in range(10)]),[100,1])
            a = a = np.reshape(np.array([[(e/4.5 - 1.)] for e in range(10) for _ in range(10)]),[10,10]).T
            b = np.reshape(a,[100,1])
            c = np.zeros_like(b)
            lcont_sample = np.hstack([b,c])
            samples = sess.run(Gz,feed_dict={z_in:z_sample,latent_cat_in:lcat_sample,latent_cont_in:lcont_sample}) #Use new z to get sample images from generator.
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            #Save sample generator images for viewing training progress.
            save_images(np.reshape(samples[0:100],[100,32,32]),[10,10],sample_directory+'/fig'+str(i)+'.png')
        if i % 1000 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            print ("Saved Model")
