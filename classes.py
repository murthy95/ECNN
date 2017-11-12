import numpy as np
import os
from scipy import misc
import keras

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

#computing the gradient map for combined images
  def get_gradient(self, img):
      i = np.zeros(img.shape)
      i += np.array(img)

      i_left = np.zeros(img.shape)
      i_right = np.zeros(img.shape)
      i_up = np.zeros(img.shape)
      i_down = np.zeros(img.shape)

      h = img.shape[0]
      w = img.shape[1]

      i_left[:,0:w-1,:] = i[:,1:w,:]
      i_right[:,1:w,:] = i[:,0:w-1,:]
      i_up[0:h-1,:,:] = i[1:h,:,:]
      i_down[1:h,:,:] = i[0:h-1,:,:]

      i_sum = abs(i-i_left) + abs(i-i_right) + abs(i-i_up) + abs(i-i_down)
      i_sum = np.sum(i_sum,axis=2)
      i_sum /= 4

      return i_sum.reshape((i_sum.shape[0], i_sum.shape[1], 1))

  def __data_generation(self, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
      y = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
        #   X[i, :, :, :, 0] = np.load(ID + '.npy')


          # Store class
        #   y[i] = labels[ID]
          ID = 'Data'+ ID.split('.')[1]
          x_combined = misc.imread(str(ID)+'.jpg')
          x_ref = misc.imread(str(ID)+'_r.jpg')
          x_bg = misc.imread(str(ID)+'_b.jpg')
          x_grad = self.get_gradient(x_combined)
          x_ref_grad = self.get_gradient(x_ref)
          x_bg_grad = self.get_gradient(x_bg)
        #   print x_combined.shape
        #   print x_grad.shape
          x = np.concatenate((x_combined, x_grad), axis=2)

          #can include gradients of reflection also here
          X[i] = x/255
          y[i] = x_ref_grad/255

      return X, y


class training_log(keras.callbacks.Callback):
    def __init__(self):
        self.train_loss=[]
        self.val_loss=[]
        self.epoch_counter=0

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.epoch_counter +=1
        file_path = 'model_ref_epoch'+str(self.epoch_counter)+'.h5'
        self.model.save_weights(file_path)
