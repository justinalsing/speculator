__all__ = (ParameterEstimator)

import numpy as np
import tensorflow as tf
import pickle
import tensorflow_probability as tfp
dtype = tf.float32
tfb = tfp.bijectors


class ParameterEstimator(tf.keras.Model):
    """
    ParameterEstimator model
    """

    def __init__(self, n_parameters=None, n_inputs=None, parameters_upper=None, parameters_lower=None, inputs_shift=None, inputs_scale=None, n_hidden=[50,50], restore=False, restore_filename=None, optimizer=tf.keras.optimizers.Adam()):
        
        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_inputs: number of inputs to input
        :param parameters_lower: lower limits for parameters
        :param parameters_upper: upper limits for parameters
        :param inputs_shift: shift for the input mags
        :param inputs_scale: scale for the input mags
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param restore: (bool) whether to restore an previously trained model or not
        :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
        """
        
        # super
        super(ParameterEstimator, self).__init__()
        
        # restore
        if restore is True:
            self.restore(restore_filename)
            
        # else set variables from input parameters
        else:
            # parameters
            self.n_parameters = n_parameters
            self.n_hidden = n_hidden
            self.n_inputs = n_inputs

            # architecture
            self.architecture = [self.n_inputs] + self.n_hidden + [self.n_parameters]
            self.n_layers = len(self.architecture) - 1

            # shifts and scales...
        
            # input parameters shift and scale
            self.parameters_lower_ = parameters_lower if parameters_lower is not None else np.zeros(self.n_parameters)
            self.parameters_upper_ = parameters_upper if parameters_upper is not None else np.ones(self.n_parameters)

            # spectrum shift and scale
            self.inputs_shift_ = inputs_shift if inputs_shift is not None else np.zeros(self.n_inputs)
            self.inputs_scale_ = inputs_scale if inputs_scale is not None else np.ones(self.n_inputs)

        # shifts and scales and transform matrix into tensorflow constants...
        
        # input parameters shift and scale
        self.parameters_lower = tf.constant(self.parameters_lower_, dtype=dtype, name='parameters_lower')
        self.parameters_upper = tf.constant(self.parameters_upper_, dtype=dtype, name='parameters_upper')
        
        # spectrum shift and scale
        self.inputs_shift = tf.constant(self.inputs_shift_, dtype=dtype, name='inputs_shift')
        self.inputs_scale = tf.constant(self.inputs_scale_, dtype=dtype, name='inputs_scale')
        
        # trainable variables...
        
        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = [] 
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-10), name="W_" + str(i)))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i)))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i)))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i)))
        
        # restore weights if restore = True
        if restore is True:
            for i in range(self.n_layers):
                self.W[i].assign(self.W_[i])
                self.b[i].assign(self.b_[i])
            for i in range(self.n_layers-1):
                self.alphas[i].assign(self.alphas_[i])
                self.betas[i].assign(self.betas_[i])

        # optimizer
        self.optimizer = optimizer

        # create bijector
        self.ranges = self.parameters_upper - self.parameters_lower
        self.central = self.parameters_lower + self.ranges/2.
        if self.n_parameters > 1:
          self.bijector = tfb.Blockwise([tfb.Invert(tfb.Chain([tfb.Invert(tfb.Tanh()), tfb.Scale(2./self.ranges[i]), tfb.Shift(-self.central[i])])) for i in range(self.n_parameters)])
        else:
          self.bijector = tfb.Invert(tfb.Chain([tfb.Invert(tfb.Tanh()), tfb.Scale(2./self.ranges), tfb.Shift(-self.central)]))

    # non-linear activation function
    def activation(self, x, alpha, beta):
        
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict inputs
    @tf.function
    def call(self, inputs):
        
        outputs = []
        layers = [tf.divide(tf.subtract(inputs, self.inputs_shift), self.inputs_scale)]
        for i in range(self.n_layers - 1):
            
            # linear network operation
            outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))
            
            # non-linear activation function
            layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

        # linear output layer
        layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))
            
        # pass the outputs through the bijector
        return self.bijector(layers[-1])
            
    # pass inputs through the network to predict spectrum
    def parameters(self, inputs):
        
        # call forward pass through network
        return self.call(inputs)
    
    # save network parameters to numpy arrays
    def update_emulator_parameters(self):
        
        # put network parameters to numpy arrays
        self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]
        
        # put shift and scale parameters to numpy arrays
        self.parameters_upper_ = self.parameters_upper.numpy()
        self.parameters_lower_ = self.parameters_lower.numpy()
        self.inputs_shift_ = self.inputs_shift.numpy()
        self.inputs_scale_ = self.inputs_scale.numpy()
        
    # save
    def save(self, filename):
 
        # attributes
        attributes = [self.W_, 
                      self.b_, 
                      self.alphas_, 
                      self.betas_, 
                      self.parameters_upper_, 
                      self.parameters_lower_,
                      self.inputs_shift_,
                      self.inputs_scale_,
                      self.n_parameters,
                      self.n_inputs,
                      self.n_hidden,
                      self.n_layers,
                      self.architecture]
        
        # save attributes to file
        f = open(filename + ".pkl", 'wb')
        pickle.dump(attributes, f)
        f.close()
        
    # restore attributes
    def restore(self, filename):
        
        # load attributes
        f = open(filename + ".pkl", 'rb')
        self.W_, self.b_, self.alphas_, self.betas_, self.parameters_upper_, \
        self.parameters_lower_, self.inputs_shift_, \
        self.inputs_scale_, self.n_parameters, self.n_inputs, \
        self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
        f.close()

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss(self, inputs, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(inputs), parameters)))      

    @tf.function
    def compute_loss_and_gradients(self, inputs, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(inputs), parameters)))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step(self, inputs, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients(inputs, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients(self, inputs, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((inputs, parameters)).batch(int(inputs.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for inputs_, parameters_ in dataset:
        
        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients(inputs_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*inputs_.shape[0]/inputs.shape[0])
        accumulated_loss.assign_add(loss*inputs_.shape[0]/inputs.shape[0])
        
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss


