import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

import argparse
import numpy as np
import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec

from skimage.util import view_as_blocks
from scipy.io import loadmat
from scipy.linalg import norm,orth


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Model Properties
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=100,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')

# Optimization
parser.add_argument('-l', '--step_size', type=float,
                    default=20.0, help='Step size for optimization')
parser.add_argument('-p', '--p_dropout', type=float, default=0.2,
                    help='Pixel dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=2001, help='How many epochs to run in total?')
parser.add_argument('-M', '--momentum', type=float, default=0.9,
                    help= 'Momentum')

# Evaluation
parser.add_argument('-E', '--expt_name', type=str, default='default',
                    help= 'Saves images in a directory with this name')
parser.add_argument('-H', '--mr', type=float, default= 0.25,
                    help= 'Measurement Rate')
parser.add_argument('-Z', '--size', type=int, default= 128,
                    help= 'Image size')
parser.add_argument('-L', '--load', type=str, default= 'images/parrot_cropped.jpg',
                    help= 'Loading a custom image')
parser.add_argument('-C', '--save_freq',type=int, default=100,
                    help= 'Reconstructed images saved every save_freq iterations')
parser.add_argument('-R', '--print_freq',type=int, default=10,
                    help= 'Optimization details printed every print_freq iterations')
parser.add_argument('-o', '--save_dir', type=str, default='spc',
                    help='Location for results')
parser.add_argument('-g', '--nr_gpu', type=int, default=1,
                    help='How many GPUs to distribute the training across?')
parser.add_argument('-P', '--gpu_num', type=str, default="-1",
                    help='GPU index')
parser.add_argument('-s', '--seed', type=int, default=2,
                    help='Random seed to use')
args = parser.parse_args()


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
if args.gpu_num != "-1":
     os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num 
import tensorflow as tf 

path = args.save_dir+'/'+args.expt_name+'/'
if not os.path.exists(path):
     os.makedirs(path)
N = args.size
m = int(args.mr*args.size**2)
batch_size = (args.size / 64)**2 / args.nr_gpu


#Loading data
test_img = mplimg.imread(args.load)[:args.size, -args.size:, :]
tile_shape = (args.size/64,args.size/64)
test_imgs = np.zeros((args.nr_gpu*batch_size,64,64,3))
for i in range(3):
     test_imgs[:,:,:,i] =view_as_blocks(test_img[:,:,i],(64,64)).reshape(-1,64,64)


#Computational Graph
xs = [tf.placeholder(tf.float32,shape=(batch_size,64,64,3)) for i in range(args.nr_gpu)]
x_init = tf.placeholder(tf.float32,shape=(batch_size,64,64,3))
model_opt = {'nr_resnet':args.nr_resnet, 'nr_filters': args.nr_filters,'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity':args.resnet_nonlinearity}
model = tf.make_template('model',model_spec)

gen_par_init = model(x_init,None,init=True,ema=None,dropout_p=0.,**model_opt)

grads = []
loss_gen = []
for i in range(args.nr_gpu):
     with tf.device('/gpu:%d' % i):
          gen_par = model(xs[i],None,ema=None,dropout_p=0.,**model_opt)
          loss,prior = nn.discretized_mix_logistic_loss(xs[i],gen_par)
          loss_gen.append(loss)
          grads.append(tf.gradients(loss_gen,[xs[i]]))

with tf.device('/gpu:0'):
     for i in range(1, args.nr_gpu):
          loss_gen[0] += loss_gen[i]
     loss_gen_sum = loss_gen[0]/(args.nr_gpu* np.log(2.)* np.prod([64,64,3])* batch_size)
     grads_sum = tf.squeeze(tf.concat(grads,axis=1))/(args.nr_gpu* np.log(2.)* np.prod([64,64,3])* batch_size)

saver = tf.train.Saver()
sess = tf.Session()


print 'Restoring weights'
saver.restore(sess,'saves/params_imagenet_epoch6.ckpt') 


print 'Creating Phi and measurments'
np.random.seed(args.seed)
Phi = np.zeros((m,N**2,3))
for i in range(3):
     Phi[:,:,i] = orth(np.random.randn(m,N**2).T).T
test_img_r = np.cast[np.float64](test_img/255.0)

y = np.einsum('mnr,ndr->mdr',Phi,test_img_r.reshape(-1,1,3))


#Initialization
np.random.seed(3)
init_imgs =np.random.uniform(low=0.0,high=255.0,size=test_imgs.shape)
init_imgs = np.cast[np.float32]((init_imgs- 127.5) / 127.5)
out_img = plotting.img_tile(test_imgs,border=0,stretch = True,tile_shape=tile_shape)
mplimg.imsave(path+'/original_img',out_img)
prev_update = np.zeros_like(init_imgs)
l=0


print 'Optimization started'
for itr in range(args.max_epochs):
     # Backpropagation to inputs
     x = np.split(init_imgs,args.nr_gpu)
     grads_n,loss_gen_n = sess.run([grads_sum,loss_gen_sum],{xs[i]: x[i] for i in range(args.nr_gpu)})     
     select = np.random.choice(2,grads_n.shape,p=[args.p_dropout,1 - args.p_dropout])

     # Gradient ascent 
     current_update = args.step_size*select*grads_n + args.momentum*prev_update
     init_imgs -= current_update 
     prev_update = current_update

     # Stitching
     init_img = plotting.img_tile(init_imgs,border=0,stretch=True,tile_shape=tile_shape)

     # Projection 
     l = 0
     res = y - np.einsum('mnr,ndr->mdr',Phi,init_img.reshape(-1,1,3))
     l += norm(res)
     init_img += np.einsum('mnr,ndr->mdr',Phi.transpose([1,0,2]),res).reshape(args.size,args.size,3)
     l = l/3

     # Clipping 
     init_img = np.clip(init_img,0.0,1.0)

     # Splitting
     for i in range(3): 
          init_imgs[:,:,:,i] =2*view_as_blocks(init_img[:,:,i],(64,64)).reshape(-1,64,64)-1

     # Printing
     if itr % args.print_freq == 0:
          print 'Iter: ', itr, '- Log Prior', loss_gen_n, '- Log Likelihood', l

     # Saving images
     if itr % args.save_freq == 0:
           mplimg.imsave(path+'/img'+str(itr+1),init_img)
     