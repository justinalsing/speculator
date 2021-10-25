import numpy as np
import tensorflow as tf
import pickle
from sklearn.decomposition import IncrementalPCA
import tqdm

dtype = tf.float32

class Speculator(tf.keras.Model):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, pca_transform_matrix=None, parameters_shift=None, parameters_scale=None, pca_shift=None, pca_scale=None, log_spectrum_shift=None, log_spectrum_scale=None, n_hidden=[50,50], restore=False, restore_filename=None, trainable=True, optimizer=tf.keras.optimizers.Adam()):

        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param pca_transform_matrix: the PCA basis vectors, ie., an [n_pcas x n_wavelengths] matrix
        :param parameters_shift: shift for input parameters
        :param parameters_scalet: scale for input parameters
        :param pca_shift: shift for PCA coefficients
        :param pca_scale: scale for PCA coefficients
        :param log_spectrum_shift: shift for the output spectra
        :param log_spectrum_scale: scale for the output spectra
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param restore: (bool) whether to restore an previously trained model or not
        :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
        """

        # super
        super(Speculator, self).__init__()

        # restore
        if restore is True:
            self.restore(restore_filename)

        # else set variables from input parameters
        else:
            # parameters
            self.n_parameters = n_parameters
            self.n_wavelengths = pca_transform_matrix.shape[-1]
            self.n_pcas = pca_transform_matrix.shape[0]
            self.n_hidden = n_hidden
            self.wavelengths = wavelengths

            # architecture
            self.architecture = [self.n_parameters] + self.n_hidden + [self.n_pcas]
            self.n_layers = len(self.architecture) - 1

            # PCA transform matrix
            self.pca_transform_matrix_ = pca_transform_matrix

            # shifts and scales...

            # input parameters shift and scale
            self.parameters_shift_ = parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters)
            self.parameters_scale_ = parameters_scale if parameters_scale is not None else np.ones(self.n_parameters)

            # PCA shift and scale
            self.pca_shift_ = pca_shift if pca_shift is not None else np.zeros(self.n_pcas)
            self.pca_scale_ = pca_scale if pca_scale is not None else np.ones(self.n_pcas)

            # spectrum shift and scale
            self.log_spectrum_shift_ = log_spectrum_shift if log_spectrum_shift is not None else np.zeros(self.n_wavelengths)
            self.log_spectrum_scale_ = log_spectrum_scale if log_spectrum_scale is not None else np.ones(self.n_wavelengths)

        # shifts and scales and transform matrix into tensorflow constants...

        # input parameters shift and scale
        self.parameters_shift = tf.constant(self.parameters_shift_, dtype=dtype, name='parameters_shift')
        self.parameters_scale = tf.constant(self.parameters_scale_, dtype=dtype, name='parameters_scale')

        # PCA shift and scale
        self.pca_shift = tf.constant(self.pca_shift_, dtype=dtype, name='pca_shift')
        self.pca_scale = tf.constant(self.pca_scale_, dtype=dtype, name='pca_scale')

        # spectrum shift and scale
        self.log_spectrum_shift = tf.constant(self.log_spectrum_shift_, dtype=dtype, name='log_spectrum_shift')
        self.log_spectrum_scale = tf.constant(self.log_spectrum_scale_, dtype=dtype, name='log_spectrum_scale')

        # pca transform matrix
        self.pca_transform_matrix = tf.constant(self.pca_transform_matrix_, dtype=dtype, name='pca_transform_matrix')

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

        # restore weights if restore = True
        if restore is True:
            for i in range(self.n_layers):
                self.W[i].assign(self.W_[i])
                self.b[i].assign(self.b_[i])
            for i in range(self.n_layers-1):
                self.alphas[i].assign(self.alphas_[i])
                self.betas[i].assign(self.betas_[i])

        self.optimizer = optimizer

    # non-linear activation function
    def activation(self, x, alpha, beta):

        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict pca coefficients
    @tf.function
    def call(self, parameters):

        outputs = []
        layers = [tf.divide(tf.subtract(parameters, self.parameters_shift), self.parameters_scale)]
        for i in range(self.n_layers - 1):

            # linear network operation
            outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))

            # non-linear activation function
            layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

        # linear output layer
        layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))

        # rescale the output (predicted PCA coefficients) and return
        return tf.add(tf.multiply(layers[-1], self.pca_scale), self.pca_shift)

    # pass inputs through the network to predict spectrum
    @tf.function
    def log_spectrum(self, parameters):

        # pass through network to compute PCA coefficients
        pca_coefficients = self.call(parameters)

        # transform from PCA to normalized spectrum basis; shift and re-scale normalized spectrum -> spectrum
        return tf.add(tf.multiply(tf.matmul(pca_coefficients, self.pca_transform_matrix), self.log_spectrum_scale), self.log_spectrum_shift)

    # save network parameters to numpy arrays
    def update_emulator_parameters(self):

        # put network parameters to numpy arrays
        self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]

        # put shift and scale parameters to numpy arrays
        self.parameters_shift_ = self.parameters_shift.numpy()
        self.parameters_scale_ = self.parameters_scale.numpy()
        self.pca_shift_ = self.pca_shift.numpy()
        self.pca_scale_ = self.pca_scale.numpy()
        self.log_spectrum_shift_ = self.log_spectrum_shift.numpy()
        self.log_spectrum_scale_ = self.log_spectrum_scale.numpy()

        # pca transform matrix
        self.pca_transform_matrix_ = self.pca_transform_matrix.numpy()

    # save
    def save(self, filename):

        # attributes
        attributes = [self.W_,
                      self.b_,
                      self.alphas_,
                      self.betas_,
                      self.parameters_shift_,
                      self.parameters_scale_,
                      self.pca_shift_,
                      self.pca_scale_,
                      self.log_spectrum_shift_,
                      self.log_spectrum_scale_,
                      self.pca_transform_matrix_,
                      self.n_parameters,
                      self.n_wavelengths,
                      self.wavelengths,
                      self.n_pcas,
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
        self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_, \
        self.parameters_scale_, self.pca_shift_, self.pca_scale_, self.log_spectrum_shift_, \
        self.log_spectrum_scale_, self.pca_transform_matrix_, self.n_parameters, self.n_wavelengths, \
        self.wavelengths, self.n_pcas, self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
        f.close()

    # forward prediction of spectrum given input parameters implemented in numpy
    def log_spectrum_(self, parameters):

        # forward pass through the network
        act = []
        layers = [(parameters - self.parameters_shift_)/self.parameters_scale_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (normalized) PCA coefficients
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale PCA coefficients, multiply out PCA basis -> normalized spectrum, shift and re-scale spectrum -> output spectrum
        return np.dot(layers[-1]*self.pca_scale_ + self.pca_shift_, self.pca_transform_matrix_)*self.log_spectrum_scale_ + self.log_spectrum_shift_

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss_spectra(self, spectra, parameters, noise_floor):

      return tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(tf.exp(self.log_spectrum(parameters)), spectra), tf.square(noise_floor))))

    @tf.function
    def compute_loss_and_gradients_spectra(self, spectra, parameters, noise_floor):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(tf.exp(self.log_spectrum(parameters)), spectra), tf.square(noise_floor))))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_spectra(self, spectra, parameters, noise_floor):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_spectra(spectra, parameters, noise_floor)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_spectra(self, spectra, parameters, noise_floor, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((spectra, parameters, noise_floor)).batch(int(spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for spectra_, parameters_, noise_floor_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_spectra(spectra_, parameters_, noise_floor_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*spectra_.shape[0]/spectra.shape[0])
        accumulated_loss.assign_add(loss*spectra_.shape[0]/spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss

    @tf.function
    def compute_loss_pca(self, pca, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(parameters), pca)))

    @tf.function
    def compute_loss_and_gradients_pca(self, pca, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(parameters), pca)))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_pca(self, pca, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_pca(pca, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_pca(self, pca, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((pca, parameters)).batch(int(pca.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for pca_, parameters_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_pca(pca_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*pca_.shape[0]/pca.shape[0])
        accumulated_loss.assign_add(loss*pca_.shape[0]/pca.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss

    @tf.function
    def compute_loss_log_spectra(self, log_spectra, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

    @tf.function
    def compute_loss_and_gradients_log_spectra(self, log_spectra, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_log_spectra(self, log_spectra, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_log_spectra(self, log_spectra, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((log_spectra, parameters)).batch(int(log_spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for log_spectra_, parameters_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*log_spectra_.shape[0]/log_spectra.shape[0])
        accumulated_loss.assign_add(loss*log_spectra_.shape[0]/log_spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss


class SpectrumPCA():
    """
    SPECULATOR PCA compression class
    """

    def __init__(self, n_parameters, n_wavelengths, n_pcas, log_spectrum_filenames, parameter_filenames, parameter_selection = None):
        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param n_pcas: number of PCA components
        :param log_spectrum_filenames: list of .npy filenames for log spectra (each one an [n_samples, n_wavelengths] array)
        :param parameter_filenames: list of .npy filenames for parameters (each one an [n_samples, n_parameters] array)
        """

        # input parameters
        self.n_parameters = n_parameters
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.log_spectrum_filenames = log_spectrum_filenames
        self.parameter_filenames = parameter_filenames
        self.n_batches = len(self.parameter_filenames)

        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)

        # parameter selection (implementing any cuts on strange parts of parameter space)
        self.parameter_selection = parameter_selection

    # compute shift and scale for spectra and parameters
    def compute_spectrum_parameters_shift_and_scale(self):

        # shift and scale
        self.log_spectrum_shift = np.zeros(self.n_wavelengths)
        self.log_spectrum_scale = np.zeros(self.n_wavelengths)
        self.parameter_shift = np.zeros(self.n_parameters)
        self.parameter_scale = np.zeros(self.n_parameters)

        # loop over training data files, accumulate means and std deviations
        for i in range(self.n_batches):

            # accumulate assuming no parameter selection
            if self.parameter_selection is None:
                self.log_spectrum_shift += np.mean(np.load(self.log_spectrum_filenames[i]), axis=0)/self.n_batches
                self.log_spectrum_scale += np.std(np.load(self.log_spectrum_filenames[i]), axis=0)/self.n_batches
                self.parameter_shift += np.mean(np.load(self.parameter_filenames[i]), axis=0)/self.n_batches
                self.parameter_scale += np.std(np.load(self.parameter_filenames[i]), axis=0)/self.n_batches
            # else make selections and accumulate
            else:
                # import spectra and make parameter-based cut
                log_spectra = np.load(self.log_spectrum_filenames[i])
                parameters = np.load(self.parameter_filenames[i])
                selection = self.parameter_selection(parameters)

                # update shifts and scales
                self.log_spectrum_shift += np.mean(log_spectra[selection,:], axis=0)/self.n_batches
                self.log_spectrum_scale += np.std(log_spectra[selection,:], axis=0)/self.n_batches
                self.parameter_shift += np.mean(parameters[selection,:], axis=0)/self.n_batches
                self.parameter_scale += np.std(parameters[selection,:], axis=0)/self.n_batches

    # train PCA incrementally
    def train_pca(self):

        # loop over training data files, increment PCA
        for i in range(self.n_batches):

            if self.parameter_selection is None:

                # load spectra and shift+scale
                normalized_log_spectra = (np.load(self.log_spectrum_filenames[i]) - self.log_spectrum_shift)/self.log_spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_log_spectra)

            else:

                # select based on parameters
                selection = self.parameter_selection(np.load(self.parameter_filenames[i]))

                # load spectra and shift+scale
                normalized_log_spectra = (np.load(self.log_spectrum_filenames[i])[selection,:] - self.log_spectrum_shift)/self.log_spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_log_spectra)

        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_

    # transform the training data set to PCA basis
    def transform_and_stack_training_data(self, filename, retain = False):

        # transform the spectra to PCA basis
        training_pca = np.concatenate([self.PCA.transform((np.load(self.log_spectrum_filenames[i]) - self.log_spectrum_shift)/self.log_spectrum_scale) for i in range(self.n_batches)])

        # stack the input parameters
        training_parameters = np.concatenate([np.load(self.parameter_filenames[i]) for i in range(self.n_batches)])

        if self.parameter_selection is not None:
            selection = self.parameter_selection(training_parameters)
            training_pca = training_pca[selection,:]
            training_parameters = training_parameters[selection,:]

        # shift and scale of PCA basis
        self.pca_shift = np.mean(training_pca, axis=0)
        self.pca_scale = np.std(training_pca, axis=0)

        # save stacked transformed training data
        np.save(filename + '_pca.npy', training_pca)
        np.save(filename + '_parameters.npy', training_parameters)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters

    # make a validation plot of the PCA given some validation data
    def validate_pca_basis(self, log_spectrum_filename, parameter_filename=None):

        # load in the data (and select based on parameter selection if neccessary)
        if self.parameter_selection is None:

            # load spectra and shift+scale
            log_spectra = np.load(log_spectrum_filename)
            normalized_log_spectra = (log_spectra - self.log_spectrum_shift)/self.log_spectrum_scale

        else:

            # select based on parameters
            selection = self.parameter_selection(np.load(self.parameter_filename))

            # load spectra and shift+scale
            log_spectra = np.load(log_spectrum_filename)[selection,:]
            normalized_log_spectra = (log_spectra - self.log_spectrum_shift)/self.log_spectrum_scale

        # transform to PCA basis and back
        log_spectra_pca = self.PCA.transform(normalized_log_spectra)
        log_spectra_in_basis = np.dot(log_spectra_pca, self.pca_transform_matrix)*self.log_spectrum_scale + self.log_spectrum_shift

        # return raw spectra and spectra in basis
        return log_spectra, log_spectra_in_basis


class Photulator(tf.keras.Model):
    """
    PHOTULATOR model
    """

    def __init__(self, n_parameters=None, filters=None, parameters_shift=None, parameters_scale=None, magnitudes_shift=None, magnitudes_scale=None, n_hidden=[50,50], restore=False, restore_filename=None, trainable=True, optimizer=tf.keras.optimizers.Adam()):

        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param filters: list of filter names
        :param parameters_shift: shift for input parameters
        :param parameters_scale: scale for input parameters
        :param magnitudes_shift: shift for the output mags
        :param magnitudes_scale: scale for the output mags
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param restore: (bool) whether to restore an previously trained model or not
        :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
        """

        # super
        super(Photulator, self).__init__()

        # restore
        if restore is True:
            self.restore(restore_filename)

        # else set variables from input parameters
        else:
            # parameters
            self.n_parameters = n_parameters
            self.n_hidden = n_hidden
            self.filters = filters
            self.n_filters = len(filters)

            # architecture
            self.architecture = [self.n_parameters] + self.n_hidden + [self.n_filters]
            self.n_layers = len(self.architecture) - 1

            # shifts and scales...

            # input parameters shift and scale
            self.parameters_shift_ = parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters)
            self.parameters_scale_ = parameters_scale if parameters_scale is not None else np.ones(self.n_parameters)

            # spectrum shift and scale
            self.magnitudes_shift_ = magnitudes_shift if magnitudes_shift is not None else np.zeros(self.n_filters)
            self.magnitudes_scale_ = magnitudes_scale if magnitudes_scale is not None else np.ones(self.n_filters)

        # shifts and scales and transform matrix into tensorflow constants...

        # input parameters shift and scale
        self.parameters_shift = tf.constant(self.parameters_shift_, dtype=dtype, name='parameters_shift')
        self.parameters_scale = tf.constant(self.parameters_scale_, dtype=dtype, name='parameters_scale')

        # spectrum shift and scale
        self.magnitudes_shift = tf.constant(self.magnitudes_shift_, dtype=dtype, name='magnitudes_shift')
        self.magnitudes_scale = tf.constant(self.magnitudes_scale_, dtype=dtype, name='magnitudes_scale')

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

        # restore weights if restore = True
        if restore is True:
            for i in range(self.n_layers):
                self.W[i].assign(self.W_[i])
                self.b[i].assign(self.b_[i])
            for i in range(self.n_layers-1):
                self.alphas[i].assign(self.alphas_[i])
                self.betas[i].assign(self.betas_[i])

        self.optimizer = optimizer

    # non-linear activation function
    def activation(self, x, alpha, beta):

        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict magnitudes
    @tf.function
    def call(self, parameters):

        outputs = []
        layers = [tf.divide(tf.subtract(parameters, self.parameters_shift), self.parameters_scale)]
        for i in range(self.n_layers - 1):

            # linear network operation
            outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))

            # non-linear activation function
            layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

        # linear output layer
        layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))

        # rescale the output and return
        return tf.add(tf.multiply(layers[-1], self.magnitudes_scale), self.magnitudes_shift)

    # pass inputs through the network to predict spectrum
    @tf.function
    def magnitudes(self, parameters):

        # call forward pass through network
        return self.call(parameters)

    # save network parameters to numpy arrays
    def update_emulator_parameters(self):

        # put network parameters to numpy arrays
        self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]

        # put shift and scale parameters to numpy arrays
        self.parameters_shift_ = self.parameters_shift.numpy()
        self.parameters_scale_ = self.parameters_scale.numpy()
        self.magnitudes_shift_ = self.magnitudes_shift.numpy()
        self.magnitudes_scale_ = self.magnitudes_scale.numpy()

    # save
    def save(self, filename):

        # attributes
        attributes = [self.W_,
                      self.b_,
                      self.alphas_,
                      self.betas_,
                      self.parameters_shift_,
                      self.parameters_scale_,
                      self.magnitudes_shift_,
                      self.magnitudes_scale_,
                      self.n_parameters,
                      self.n_filters,
                      self.filters,
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
        self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_, \
        self.parameters_scale_, self.magnitudes_shift_, \
        self.magnitudes_scale_, self.n_parameters, self.n_filters, self.filters, \
        self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
        f.close()

    # forward prediction of spectrum given input parameters implemented in numpy
    def magnitudes_(self, parameters):

        # forward pass through the network
        act = []
        layers = [(parameters - self.parameters_shift_)/self.parameters_scale_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (normalized) PCA coefficients
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale and output
        return layers[-1]*self.magnitudes_scale_ + self.magnitudes_shift_

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss(self, theta, mags):

        return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(theta), mags)))

    @tf.function
    def compute_loss_and_gradients(self, theta, mags):

        # compute loss on the tape
        with tf.GradientTape() as tape:

            # loss
            loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.call(theta), mags)))

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        return loss, gradients

    def training_step(self, theta, mags):

        # compute loss and gradients
        loss, gradients = self.compute_loss_and_gradients(theta, mags)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def training_step_with_accumulated_gradients(self, theta, mags, accumulation_steps=10):

        # create dataset to do sub-calculations over
        dataset = tf.data.Dataset.from_tensor_slices((theta, mags)).batch(int(theta.shape[0]/accumulation_steps))

        # initialize gradients and loss (to zero)
        accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
        accumulated_loss = tf.Variable(0., trainable=False)

        # loop over sub-batches
        for theta_, mags_ in dataset:

            # calculate loss and gradients
            loss, gradients = self.compute_loss_and_gradients(theta_, mags_)

            # update the accumulated gradients and loss
            for i in range(len(accumulated_gradients)):
                accumulated_gradients[i].assign_add(gradients[i]*theta_.shape[0]/theta.shape[0])
            accumulated_loss.assign_add(loss*theta_.shape[0]/theta.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

        return accumulated_loss



class SpeculatorLogAutoencoder(tf.keras.Model):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, parameters_shift=None, parameters_scale=None, log_spectrum_shift=None, log_spectrum_scale=None, n_hidden=[50,50], optimizer=tf.keras.optimizers.Adam(), restore=False, restore_filename=None, trainable=True):

      """
      Constructor.
      :param n_parameters: number of SED model parameters (inputs to the network)
      :param n_wavelengths: number of wavelengths in the modelled SEDs
      :param parameters_shift: shift for input parameters
      :param parameters_scalet: scale for input parameters
      :param log_spectrum_shift: shift for the output spectra
      :param log_spectrum_scale: scale for the output spectra
      :param n_hiddens: list with number of hidden units for each hidden layer
      :param restore: (bool) whether to restore an previously trained model or not
      :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
      """

      # super
      super(SpeculatorLogAutoencoder, self).__init__()

      # restore
      if restore is True:
        self.restore(restore_filename)

      # else set variables from input parameters
      else:
        # parameters
        self.n_parameters = n_parameters
        self.wavelengths = wavelengths
        self.n_wavelengths = len(wavelengths)
        self.n_hidden = n_hidden
        self.wavelengths = wavelengths

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_wavelengths]
        self.n_layers = len(self.architecture) - 1

        # shifts and scales...

        # input parameters shift and scale
        self.parameters_shift_ = parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters)
        self.parameters_scale_ = parameters_scale if parameters_scale is not None else np.ones(self.n_parameters)

        # spectrum shift and scale
        self.log_spectrum_shift_ = log_spectrum_shift if log_spectrum_shift is not None else np.zeros(self.n_wavelengths)
        self.log_spectrum_scale_ = log_spectrum_scale if log_spectrum_scale is not None else np.ones(self.n_wavelengths)

      # shifts and scales and transform matrix into tensorflow constants...

      # input parameters shift and scale
      self.parameters_shift = tf.constant(self.parameters_shift_, dtype=dtype, name='parameters_shift')
      self.parameters_scale = tf.constant(self.parameters_scale_, dtype=dtype, name='parameters_scale')

      # spectrum shift and scale
      self.log_spectrum_shift = tf.constant(self.log_spectrum_shift_, dtype=dtype, name='spectrum_shift')
      self.log_spectrum_scale = tf.constant(self.log_spectrum_scale_, dtype=dtype, name='spectrum_scale')

      # trainable variables...

      # weights, biases and activation function parameters for each layer of the network
      self.W = []
      self.b = []
      self.alphas = []
      self.betas = []
      for i in range(self.n_layers):
        self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3), name="W_" + str(i), trainable=trainable))
        self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
      for i in range(self.n_layers-1):
        self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
        self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

      # restore weights if restore = True
      if restore is True:
        for i in range(self.n_layers):
          self.W[i].assign(self.W_[i])
          self.b[i].assign(self.b_[i])
        for i in range(self.n_layers-1):
          self.alphas[i].assign(self.alphas_[i])
          self.betas[i].assign(self.betas_[i])

      self.optimizer = optimizer

    # non-linear activation function
    def activation(self, x, alpha, beta):

      return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict the spectra
    @tf.function
    def call(self, parameters):

      outputs = []
      layers = [tf.divide(tf.subtract(parameters, self.parameters_shift), self.parameters_scale)]
      for i in range(self.n_layers - 1):

        # linear network operation
        outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))

        # non-linear activation function
        layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

      # linear output layer: at this point we have
      layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))

      # rescale -> output log spectra
      return tf.add(tf.multiply(layers[-1], self.log_spectrum_scale), self.log_spectrum_shift)

    # spectrum
    @tf.function
    def spectrum(self, parameters):

      return tf.exp(self.call(parameters))

    # spectrum
    @tf.function
    def log_spectrum(self, parameters):

      return self.call(parameters)

    # save network parameters to numpy arrays
    def update_emulator_parameters(self):

      # put network parameters to numpy arrays
      self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
      self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
      self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
      self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]

      # put shift and scale parameters to numpy arrays
      self.parameters_shift_ = self.parameters_shift.numpy()
      self.parameters_scale_ = self.parameters_scale.numpy()
      self.log_spectrum_shift_ = self.log_spectrum_shift.numpy()
      self.log_spectrum_scale_ = self.log_spectrum_scale.numpy()

    # save
    def save(self, filename):

      # attributes
      attributes = [self.W_,
                    self.b_,
                    self.alphas_,
                    self.betas_,
                    self.parameters_shift_,
                    self.parameters_scale_,
                    self.log_spectrum_shift_,
                    self.log_spectrum_scale_,
                    self.n_parameters,
                    self.n_wavelengths,
                    self.wavelengths,
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
      self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_, \
      self.parameters_scale_, self.log_spectrum_shift_, \
      self.log_spectrum_scale_, self.n_parameters, self.n_wavelengths, \
      self.wavelengths, self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
      f.close()

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss_spectra(self, spectra, parameters, noise_floor):

      return tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(self.spectrum(parameters), spectra), tf.square(noise_floor))))

    @tf.function
    def compute_loss_and_gradients_spectra(self, spectra, parameters, noise_floor):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(self.spectrum(parameters), spectra), tf.square(noise_floor))))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_spectra(self, spectra, parameters, noise_floor):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients(spectra, parameters, noise_floor)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_spectra(self, spectra, parameters, noise_floor, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((spectra, parameters, noise_floor)).batch(int(spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for spectra_, parameters_, noise_floor_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_spectra(spectra_, parameters_, noise_floor_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*spectra_.shape[0]/spectra.shape[0])
        accumulated_loss.assign_add(loss*spectra_.shape[0]/spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss

    @tf.function
    def compute_loss_log_spectra(self, log_spectra, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

    @tf.function
    def compute_loss_and_gradients_log_spectra(self, log_spectra, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_log_spectra(self, log_spectra, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_log_spectra(self, log_spectra, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((log_spectra, parameters)).batch(int(log_spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for log_spectra_, parameters_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*log_spectra_.shape[0]/log_spectra.shape[0])
        accumulated_loss.assign_add(loss*log_spectra_.shape[0]/log_spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss


class SpeculatorLinearAutoencoder(tf.keras.Model):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, parameters_shift=None, parameters_scale=None, log_spectrum_shift=None, log_spectrum_scale=None, n_hidden=[50,50], optimizer=tf.keras.optimizers.Adam(), restore=False, restore_filename=None, trainable=True):

      """
      Constructor.
      :param n_parameters: number of SED model parameters (inputs to the network)
      :param n_wavelengths: number of wavelengths in the modelled SEDs
      :param parameters_shift: shift for input parameters
      :param parameters_scalet: scale for input parameters
      :param log_spectrum_shift: shift for the output spectra
      :param log_spectrum_scale: scale for the output spectra
      :param n_hiddens: list with number of hidden units for each hidden layer
      :param restore: (bool) whether to restore an previously trained model or not
      :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
      """

      # super
      super(SpeculatorLinearAutoencoder, self).__init__()

      # restore
      if restore is True:
        self.restore(restore_filename)

      # else set variables from input parameters
      else:
        # parameters
        self.n_parameters = n_parameters
        self.wavelengths = wavelengths
        self.n_wavelengths = len(wavelengths)
        self.n_hidden = n_hidden
        self.wavelengths = wavelengths

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_wavelengths]
        self.n_layers = len(self.architecture) - 1

        # shifts and scales...

        # input parameters shift and scale
        self.parameters_shift_ = parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters)
        self.parameters_scale_ = parameters_scale if parameters_scale is not None else np.ones(self.n_parameters)

        # spectrum shift and scale
        self.log_spectrum_shift_ = log_spectrum_shift if log_spectrum_shift is not None else np.zeros(self.n_wavelengths)
        self.log_spectrum_scale_ = log_spectrum_scale if log_spectrum_scale is not None else np.ones(self.n_wavelengths)

      # shifts and scales and transform matrix into tensorflow constants...

      # input parameters shift and scale
      self.parameters_shift = tf.constant(self.parameters_shift_, dtype=dtype, name='parameters_shift')
      self.parameters_scale = tf.constant(self.parameters_scale_, dtype=dtype, name='parameters_scale')

      # spectrum shift and scale
      self.log_spectrum_shift = tf.constant(self.log_spectrum_shift_, dtype=dtype, name='spectrum_shift')
      self.log_spectrum_scale = tf.constant(self.log_spectrum_scale_, dtype=dtype, name='spectrum_scale')

      # trainable variables...

      # weights, biases and activation function parameters for each layer of the network
      self.W = []
      self.b = []
      self.alphas = []
      self.betas = []
      for i in range(self.n_layers):
        self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3), name="W_" + str(i), trainable=trainable))
        self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
      for i in range(self.n_layers-1):
        self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
        self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

      # restore weights if restore = True
      if restore is True:
        for i in range(self.n_layers):
          self.W[i].assign(self.W_[i])
          self.b[i].assign(self.b_[i])
        for i in range(self.n_layers-1):
          self.alphas[i].assign(self.alphas_[i])
          self.betas[i].assign(self.betas_[i])

      self.optimizer = optimizer

    # non-linear activation function
    def activation(self, x, alpha, beta):

      return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict the spectra
    @tf.function
    def call(self, parameters):

      outputs = []
      layers = [tf.divide(tf.subtract(parameters, self.parameters_shift), self.parameters_scale)]
      for i in range(self.n_layers - 1):

        # linear network operation
        outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))

        # non-linear activation function
        layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

      # output layer
      layers.append(tf.einsum('ij,jk->ijk', tf.exp(layers[-1]), tf.exp(self.W[-1])))

      # rescale -> output spectra
      return tf.reduce_sum(layers[-1], axis=1)

    # spectrum
    @tf.function
    def spectrum(self, parameters):

      return self.call(parameters)

    # spectrum
    @tf.function
    def log_spectrum(self, parameters):

      return tf.math.log(self.call(parameters))

    # save network parameters to numpy arrays
    def update_emulator_parameters(self):

      # put network parameters to numpy arrays
      self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
      self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
      self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
      self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]

      # put shift and scale parameters to numpy arrays
      self.parameters_shift_ = self.parameters_shift.numpy()
      self.parameters_scale_ = self.parameters_scale.numpy()
      self.log_spectrum_shift_ = self.log_spectrum_shift.numpy()
      self.log_spectrum_scale_ = self.log_spectrum_scale.numpy()

    # save
    def save(self, filename):

      # attributes
      attributes = [self.W_,
                    self.b_,
                    self.alphas_,
                    self.betas_,
                    self.parameters_shift_,
                    self.parameters_scale_,
                    self.log_spectrum_shift_,
                    self.log_spectrum_scale_,
                    self.n_parameters,
                    self.n_wavelengths,
                    self.wavelengths,
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
      self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_, \
      self.parameters_scale_, self.log_spectrum_shift_, \
      self.log_spectrum_scale_, self.n_parameters, self.n_wavelengths, \
      self.wavelengths, self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
      f.close()

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss_spectra(self, spectra, parameters, noise_floor):

      return tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(self.spectrum(parameters), spectra), tf.square(noise_floor))))

    @tf.function
    def compute_loss_and_gradients_spectra(self, spectra, parameters, noise_floor):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(self.spectrum(parameters), spectra), tf.square(noise_floor))))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_spectra(self, spectra, parameters, noise_floor):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients(spectra, parameters, noise_floor)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_spectra(self, spectra, parameters, noise_floor, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((spectra, parameters, noise_floor)).batch(int(spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for spectra_, parameters_, noise_floor_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_spectra(spectra_, parameters_, noise_floor_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*spectra_.shape[0]/spectra.shape[0])
        accumulated_loss.assign_add(loss*spectra_.shape[0]/spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss

    @tf.function
    def compute_loss_log_spectra(self, log_spectra, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

    @tf.function
    def compute_loss_and_gradients_log_spectra(self, log_spectra, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables)

      return loss, gradients

    def training_step_log_spectra(self, log_spectra, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

    def training_step_with_accumulated_gradients_log_spectra(self, log_spectra, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((log_spectra, parameters)).batch(int(log_spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for log_spectra_, parameters_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*log_spectra_.shape[0]/log_spectra.shape[0])
        accumulated_loss.assign_add(loss*log_spectra_.shape[0]/log_spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

      return accumulated_loss


class SpeculatorLogLinearAutoencoder(tf.keras.Model):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, parameters_shift=None, parameters_scale=None, log_spectrum_shift=None, log_spectrum_scale=None, n_hidden_log=[50,50], n_hidden_linear=[50,50], optimizer=tf.keras.optimizers.Adam(), restore=False, restore_filename_log=None, restore_filename_linear=None, trainable=True):

      """
      Constructor.
      :param n_parameters: number of SED model parameters (inputs to the network)
      :param n_wavelengths: number of wavelengths in the modelled SEDs
      :param parameters_shift: shift for input parameters
      :param parameters_scalet: scale for input parameters
      :param log_spectrum_shift: shift for the output spectra
      :param log_spectrum_scale: scale for the output spectra
      :param n_hiddens: list with number of hidden units for each hidden layer
      :param restore: (bool) whether to restore an previously trained model or not
      :param restore_filename: filename tag (without suffix) for restoring trained model from file (this will be a pickle file with all of the model attributes and weights)
      """

      # super
      super(SpeculatorLogLinearAutoencoder, self).__init__()

      # create both log and linear autoencoder models
      self.LogAutoencoder = SpeculatorLogAutoencoder(n_parameters=n_parameters, wavelengths=wavelengths, parameters_shift=parameters_shift, parameters_scale=parameters_scale, log_spectrum_shift=log_spectrum_shift, log_spectrum_scale=log_spectrum_scale, n_hidden=n_hidden_log, optimizer=tf.keras.optimizers.Adam(), restore=restore, restore_filename=restore_filename_log, trainable=True)
      self.LinearAutoencoder = SpeculatorLinearAutoencoder(n_parameters=n_parameters, wavelengths=wavelengths, parameters_shift=parameters_shift, parameters_scale=parameters_scale, log_spectrum_shift=log_spectrum_shift, log_spectrum_scale=log_spectrum_scale, n_hidden=n_hidden_linear, optimizer=tf.keras.optimizers.Adam(), restore=restore, restore_filename=restore_filename_linear, trainable=True)
      self.trainable_variables_ = self.LogAutoencoder.trainable_variables + self.LinearAutoencoder.trainable_variables

    # log spectrum
    @tf.function
    def call(self, parameters):

      # linear autoencoder term
      linear_term = self.LinearAutoencoder.spectrum(parameters)

      # log autoencoder term
      log_term = self.LogAutoencoder.spectrum(parameters)

      log_spectrum = tf.add(tf.math.log(linear_term), log_term)

      return log_spectrum

    # spectrum
    @tf.function
    def spectrum(self, parameters):

      return tf.exp(self.call(parameters))

    # spectrum
    @tf.function
    def log_spectrum(self, parameters):

      return self.call(parameters)

    # save network parameters to numpy arrays
    def update_emulator_parameters(self):

      # put network parameters to numpy arrays
      self.LogAutoencoder.update_emulator_parameters()
      self.LinearAutoencoder.update_emulator_parameters()

    # save
    def save(self, filename):

      self.LogAutoencoder.save(filename + '_log')
      self.LinearAutoencoder.save(filename + '_linear')

    # restore attributes
    def restore(self, filename):

      self.LogAutoencoder.restore(filename + '_log')
      self.LinearAutoencoder.restore(filename + '_linear')

    ### Infrastructure for network training ###

    @tf.function
    def compute_loss_spectra(self, spectra, parameters, noise_floor):

      return tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(self.spectrum(parameters), spectra), tf.square(noise_floor))))

    @tf.function
    def compute_loss_and_gradients_spectra(self, spectra, parameters, noise_floor):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.divide(tf.math.squared_difference(self.spectrum(parameters), spectra), tf.square(noise_floor))))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables_)

      return loss, gradients

    def training_step_spectra(self, spectra, parameters, noise_floor):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients(spectra, parameters, noise_floor)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables_))

      return loss

    def training_step_with_accumulated_gradients_spectra(self, spectra, parameters, noise_floor, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((spectra, parameters, noise_floor)).batch(int(spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables_]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for spectra_, parameters_, noise_floor_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_spectra(spectra_, parameters_, noise_floor_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*spectra_.shape[0]/spectra.shape[0])
        accumulated_loss.assign_add(loss*spectra_.shape[0]/spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables_))

      return accumulated_loss

    @tf.function
    def compute_loss_log_spectra(self, log_spectra, parameters):

      return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

    @tf.function
    def compute_loss_and_gradients_log_spectra(self, log_spectra, parameters):

      # compute loss on the tape
      with tf.GradientTape() as tape:

        # loss
        loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_spectrum(parameters), log_spectra)))

      # compute gradients
      gradients = tape.gradient(loss, self.trainable_variables_)

      return loss, gradients

    def training_step_log_spectra(self, log_spectra, parameters):

      # compute loss and gradients
      loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra, parameters)

      # apply gradients
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables_))

      return loss

    def training_step_with_accumulated_gradients_log_spectra(self, log_spectra, parameters, accumulation_steps=10):

      # create dataset to do sub-calculations over
      dataset = tf.data.Dataset.from_tensor_slices((log_spectra, parameters)).batch(int(log_spectra.shape[0]/accumulation_steps))

      # initialize gradients and loss (to zero)
      accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables_]
      accumulated_loss = tf.Variable(0., trainable=False)

      # loop over sub-batches
      for log_spectra_, parameters_ in dataset:

        # calculate loss and gradients
        loss, gradients = self.compute_loss_and_gradients_log_spectra(log_spectra_, parameters_)

        # update the accumulated gradients and loss
        for i in range(len(accumulated_gradients)):
          accumulated_gradients[i].assign_add(gradients[i]*log_spectra_.shape[0]/log_spectra.shape[0])
        accumulated_loss.assign_add(loss*log_spectra_.shape[0]/log_spectra.shape[0])

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables_))

      return accumulated_loss


class PhotulatorModelStack:

    def __init__(self, root_dir, filenames):

        # load emulator models
        self.emulators = [Photulator(restore=True, restore_filename=root_dir+filename, trainable=False) for filename in filenames]

        # log10 constant
        self.ln10 = tf.constant(np.log(10.), dtype=tf.float32)

        # how many emulators?
        self.n_emulators = len(filenames)

    # compute fluxes (in units of nano maggies) given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def fluxes(self, theta, N):

        return tf.concat([tf.exp( tf.multiply(tf.add(tf.multiply(-0.4, tf.add(self.emulators[i](theta), tf.expand_dims(N, -1))), 9.), self.ln10) ) for i in range(self.n_emulators)], axis=-1)

    # compute magnitudes given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def magnitudes(self, theta, N):

        return tf.concat([tf.add(self.emulators[i](theta), tf.expand_dims(N, -1)) for i in range(self.n_emulators)], axis=-1)



# train photulator model stack
def train_photulator_stack(training_theta, training_mag, parameters_shift, parameters_scale, magnitudes_shift, magnitudes_scale, n_layers=4, n_units=128, filters=None, validation_split=0.1, lr=[1e-3, 1e-4, 1e-5, 1e-6], batch_size=[1000, 10000, 50000, 1000000], gradient_accumulation_steps=[1, 1, 1, 10], epochs=1000, patience=20, root_dir='', verbose=True):

    # architecture
    n_hidden = [n_units]*n_layers

    # train each band in turn
    for f in range(len(filters)):

        if verbose is True:
            print('filter ' + filters[f] + '...')

        # construct the PHOTULATOR model
        photulator = Photulator(n_parameters=training_theta.shape[-1],
                           filters=[filters[f]],
                           parameters_shift=parameters_shift,
                           parameters_scale=parameters_scale,
                           magnitudes_shift=magnitudes_shift[f],
                           magnitudes_scale=magnitudes_scale[f],
                           n_hidden=n_hidden,
                           restore=False,
                           restore_filename=None,
                           optimizer=tf.keras.optimizers.Adam())

        # train using cooling/heating schedule for lr/batch-size
        for i in range(len(lr)):

            if verbose is True:
                print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]))

            # set learning rate
            photulator.optimizer.lr = lr[i]

            # split into validation and training sub-sets
            n_validation = int(training_theta.shape[0] * validation_split)
            n_training = training_theta.shape[0] - n_validation
            training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)
            np.save(root_dir +'trainselection_' + filters[f] + '_lr'+str(i), np.where(training_selection)[0])
            np.save(root_dir +'validselection_' + filters[f] + '_lr'+str(i), np.where(~training_selection)[0])

            # create iterable dataset (given batch size)
            train_mag = tf.expand_dims(training_mag[:,f],-1)
            training_data = tf.data.Dataset.from_tensor_slices((training_theta[training_selection], train_mag[training_selection])).shuffle(n_training).batch(batch_size[i])

            # set up training loss
            training_losses = [np.infty]
            validation_losses = [np.infty]
            best_loss = np.infty
            early_stopping_counter = 0

            # loop over epochs
            def generator():
                while early_stopping_counter < patience:
                    yield

            for _ in tqdm.tqdm(generator()):
                # loop over batches for a single epoch
                for theta, mag in training_data:

                    # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
                    if gradient_accumulation_steps[i] == 1:
                        loss = photulator.training_step(theta, mag)
                    else:
                        loss = photulator.training_step_with_accumulated_gradients(theta, mag, accumulation_steps=gradient_accumulation_steps[i])

                # compute total loss and validation loss
                training_losses.append(loss)
                validation_losses.append(photulator.compute_loss(training_theta[~training_selection], train_mag[~training_selection]).numpy())

                # early stopping condition
                if validation_losses[-1] < best_loss:
                    best_loss = validation_losses[-1]
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    photulator.update_emulator_parameters()
                    photulator.save(root_dir + 'model_{}x{}_'.format(n_layers, n_units) + filters[f])
                    if verbose is True:
                        print('Training loss = ' + str(loss.numpy()))
                        print('Validation loss = ' + str(best_loss))

            np.save(root_dir +'trainlosses_' + filters[f] + '_lr'+str(i), training_losses)
            np.save(root_dir +'validlosses_' + filters[f] + '_lr'+str(i), validation_losses)
