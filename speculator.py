import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.auto import tqdm
import pickle
dtype = tf.float32

class Speculator(tf.keras.Model):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, pca_transform_matrix=None, parameters_shift=None, parameters_scale=None, pca_shift=None, pca_scale=None, spectrum_shift=None, spectrum_scale=None, n_hidden=[50,50], restore=False, restore_filename=None):
        
        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param pca_transform_matrix: the PCA basis vectors, ie., an [n_pcas x n_wavelengths] matrix
        :param parameters_shift: shift for input parameters
        :param parameters_scalet: scale for input parameters
        :param pca_shift: shift for PCA coefficients
        :param pca_scale: scale for PCA coefficients
        :param spectrum_shift: shift for the output spectra
        :param spectrum_scale: scale for the output spectra
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
            self.spectrum_shift_ = spectrum_shift if spectrum_shift is not None else np.zeros(self.n_wavelengths)
            self.spectrum_scale_ = spectrum_scale if spectrum_scale is not None else np.ones(self.n_wavelengths)

        # shifts and scales and transform matrix into tensorflow constants...
        
        # input parameters shift and scale
        self.parameters_shift = tf.constant(self.parameters_shift_, dtype=dtype, name='parameters_shift')
        self.parameters_scale = tf.constant(self.parameters_scale_, dtype=dtype, name='parameters_scale')
        
        # PCA shift and scale
        self.pca_shift = tf.constant(self.pca_shift_, dtype=dtype, name='pca_shift')
        self.pca_scale = tf.constant(self.pca_scale_, dtype=dtype, name='pca_scale')
        
        # spectrum shift and scale
        self.spectrum_shift = tf.constant(self.spectrum_shift_, dtype=dtype, name='spectrum_shift')
        self.spectrum_scale = tf.constant(self.spectrum_scale_, dtype=dtype, name='spectrum_scale')
        
        # pca transform matrix
        self.pca_transform_matrix = tf.constant(self.pca_transform_matrix_, dtype=dtype, name='pca_transform_matrix')
        
        # trainable variables...
        
        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = [] 
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i)))
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
    def log_spectrum(self, parameters):
        
        # pass through network to compute PCA coefficients
        pca_coefficients = self.call(parameters)
        
        # transform from PCA to normalized spectrum basis; shift and re-scale normalized spectrum -> spectrum
        return tf.add(tf.multiply(tf.matmul(pca_coefficients, self.pca_transform_matrix), self.spectrum_scale), self.spectrum_shift)
    
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
        self.spectrum_shift_ = self.spectrum_shift.numpy()
        self.spectrum_scale_ = self.spectrum_scale.numpy()
        
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
                      self.spectrum_shift_,
                      self.spectrum_scale_,
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
        self.parameters_scale_, self.pca_shift_, self.pca_scale_, self.spectrum_shift_, \
        self.spectrum_scale_, self.pca_transform_matrix_, self.n_parameters, self.n_wavelengths, \
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
        return np.dot(layers[-1]*self.pca_scale_ + self.pca_shift_, self.pca_transform_matrix_)*self.spectrum_scale_ + self.spectrum_shift_


class SpectrumPCA():
    """
    SPECULATOR PCA compression class
    """
    
    def __init__(self, n_parameters, n_wavelengths, n_pcas, spectrum_filenames, parameter_filenames, parameter_selection = None):
        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param n_pcas: number of PCA components
        :param spectrum_filenames: list of .npy filenames for spectra (each one an [n_samples, n_wavelengths] array)
        :param parameter_filenames: list of .npy filenames for parameters (each one an [n_samples, n_parameters] array)
        """
        
        # input parameters
        self.n_parameters = n_parameters
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.spectrum_filenames = spectrum_filenames
        self.parameter_filenames = parameter_filenames
        self.n_batches = len(self.parameter_filenames)

        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)
        
        # parameter selection (implementing any cuts on strange parts of parameter space)
        self.parameter_selection = parameter_selection

    # compute shift and scale for spectra and parameters
    def compute_spectrum_parameters_shift_and_scale(self):
        
        # shift and scale
        self.spectrum_shift = np.zeros(self.n_wavelengths)
        self.spectrum_scale = np.zeros(self.n_wavelengths)
        self.parameter_shift = np.zeros(self.n_parameters)
        self.parameter_scale = np.zeros(self.n_parameters)
        
        # loop over training data files, accumulate means and std deviations
        for i in range(self.n_batches):
            
            # accumulate assuming no parameter selection
            if self.parameter_selection is None:
                self.spectrum_shift += np.mean(np.load(self.spectrum_filenames[i]), axis=0)/self.n_batches
                self.spectrum_scale += np.std(np.load(self.spectrum_filenames[i]), axis=0)/self.n_batches
                self.parameter_shift += np.mean(np.load(self.parameter_filenames[i]), axis=0)/self.n_batches
                self.parameter_scale += np.std(np.load(self.parameter_filenames[i]), axis=0)/self.n_batches
            # else make selections and accumulate
            else:
                # import spectra and make parameter-based cut
                spectra = np.load(self.spectrum_filenames[i])
                parameters = np.load(self.parameter_filenames[i])
                selection = self.parameter_selection(parameters)
                
                # update shifts and scales
                self.spectrum_shift += np.mean(spectra[selection,:], axis=0)/self.n_batches
                self.spectrum_scale += np.std(spectra[selection,:], axis=0)/self.n_batches
                self.parameter_shift += np.mean(parameters[selection,:], axis=0)/self.n_batches
                self.parameter_scale += np.std(parameters[selection,:], axis=0)/self.n_batches             
                
    # train PCA incrementally
    def train_pca(self):
        
        # loop over training data files, increment PCA
        for i in range(self.n_batches):
            
            if self.parameter_selection is None:
            
                # load spectra and shift+scale
                normalized_spectra = (np.load(self.spectrum_filenames[i]) - self.spectrum_shift)/self.spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_spectra)
                
            else:
                
                # select based on parameters
                selection = self.parameter_selection(np.load(self.parameter_filenames[i]))
                
                # load spectra and shift+scale
                normalized_spectra = (np.load(self.spectrum_filenames[i])[selection,:] - self.spectrum_shift)/self.spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_spectra)
            
        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_
        
    # transform the training data set to PCA basis
    def transform_and_stack_training_data(self, filename, retain = False):

        # transform the spectra to PCA basis
        training_pca = np.concatenate([self.PCA.transform((np.load(self.spectrum_filenames[i]) - self.spectrum_shift)/self.spectrum_scale) for i in range(self.n_batches)])
        
        # stack the input parameters
        training_parameters = np.concatenate([np.load(self.parameter_filenames[i]) for i in range(self.n_batches)])
        
        if self.parameter_selection is not None:
            selection = self.parameter_selection(training_parameters)
            training_pca = training_pca[selection,:]
            training_parameters = training_parameters[selection,:]
            
        # shift and scale of PCA basis
        self.pca_shift = np.mean(self.training_pca, axis=0)
        self.pca_scale = np.std(self.training_pca, axis=0)
        
        # save stacked transformed training data
        np.save(filename + '_pca.npy', training_pca)
        np.save(parameter_filename + '_parameters.npy', training_parameters)
        
        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters
            
    # make a validation plot of the PCA given some validation data
    def validate_pca_basis(self, spectrum_filename, parameter_filename=None):
        
        # load in the data (and select based on parameter selection if neccessary)
        if self.parameter_selection is None:
            
            # load spectra and shift+scale
            spectra = np.load(self.spectrum_filename)
            normalized_spectra = (spectra - self.spectrum_shift)/self.spectrum_scale
                
        else:
                
            # select based on parameters
            selection = self.parameter_selection(np.load(self.parameter_filename))
                
            # load spectra and shift+scale
            spectra = np.load(self.spectrum_filename)[selection,:]
            normalized_spectra = (spectra - self.spectrum_shift)/self.spectrum_scale
        
        # transform to PCA basis and back
        spectra_pca = self.PCA.transform(normalized_spectra)
        spectra_in_basis = np.dot(spectra_pca, self.pca_transform_matrix)*self.spectrum_scale + self.spectrum_shift

        # return raw spectra and spectra in basis
        return spectra, spectra_in_basis



class Photulator(tf.keras.Model):
    """
    PHOTULATOR model
    """

    def __init__(self, n_parameters=None, filters=None, parameters_shift=None, parameters_scale=None, magnitudes_shift=None, magnitudes_scale=None, n_hidden=[50,50], restore=False, restore_filename=None):
        
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
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i)))
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


