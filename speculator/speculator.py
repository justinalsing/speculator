import numpy as np
import torch
import pickle
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader

class Speculator(torch.nn.Module):
    """
    SPECULATOR model
    """

    def __init__(self, n_parameters=None, wavelengths=None, pca_transform_matrix=None, parameters_shift=None, parameters_scale=None, pca_shift=None, pca_scale=None, log_spectrum_shift=None, log_spectrum_scale=None, n_hidden=[50,50], optimizer=lambda x: torch.optim.Adam(x, lr=1e-3), restore=False, restore_filename=None, device="cpu"):

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
        """

        # super
        super(Speculator, self).__init__()

        # parameters
        self.n_parameters = n_parameters
        self.n_wavelengths = pca_transform_matrix.shape[-1]
        self.n_pcas = pca_transform_matrix.shape[0]
        self.n_hidden = n_hidden
        self.wavelengths = wavelengths

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_pcas]
        self.n_layers = len(self.architecture) - 1


        # shifts and scales and transform matrix

        # input parameters shift and scale
        self.parameters_shift = torch.tensor(parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters), dtype=torch.float32).to(device)
        self.parameters_scale = torch.tensor(parameters_scale if parameters_scale is not None else np.ones(self.n_parameters), dtype=torch.float32).to(device)

        # PCA shift and scale
        self.pca_shift = torch.tensor(pca_shift if pca_shift is not None else np.zeros(self.n_pcas), dtype=torch.float32).to(device)
        self.pca_scale = torch.tensor(pca_scale if pca_scale is not None else np.ones(self.n_pcas), dtype=torch.float32).to(device)

        # spectrum shift and scale
        self.log_spectrum_shift = torch.tensor(log_spectrum_shift if log_spectrum_shift is not None else np.zeros(self.n_wavelengths), dtype=torch.float32).to(device)
        self.log_spectrum_scale = torch.tensor(log_spectrum_scale if log_spectrum_scale is not None else np.ones(self.n_wavelengths), dtype=torch.float32).to(device)

        # pca transform matrix
        self.pca_transform_matrix = torch.tensor(pca_transform_matrix, dtype=torch.float32).to(device)

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        for i in range(self.n_layers):
            self.W.append(torch.nn.Parameter( torch.sqrt(torch.tensor(2. / self.n_parameters)) * torch.randn((self.architecture[i], self.architecture[i+1])) ).to(device) )
            self.b.append(torch.nn.Parameter( torch.zeros((self.architecture[i+1]))).to(device))
        for i in range(self.n_layers-1):
            self.alphas.append(torch.nn.Parameter(torch.randn((self.architecture[i+1]))).to(device))
            self.betas.append(torch.nn.Parameter(torch.randn((self.architecture[i+1]))).to(device))

        self.params = torch.nn.ParameterList(self.W + self.b + self.alphas + self.betas)
        
        # optimizer
        self.optimizer_constructor = optimizer
        self.optimizer = self.optimizer_constructor(self.params)

        if restore:
            self.load_state_dict(torch.load(restore_filename))

    # change the device we're on
    def set_device(self, device):

        self.parameters_shift = self.parameters_shift.to(device)
        self.parameters_scale = self.parameters_scale.to(device)

        self.pca_shift = self.pca_shift.to(device)
        self.pca_scale = self.pca_scale.to(device)

        self.log_spectrum_shift = self.log_spectrum_shift.to(device)
        self.log_spectrum_scale = self.log_spectrum_scale.to(device)

        self.pca_transform_matrix = self.pca_transform_matrix.to(device)

        for i in range(self.n_layers):
            self.W[i] = self.W[i].to(device)
            self.b[i] = self.b[i].to(device)
        for i in range(self.n_layers-1):
            self.alphas[i] = self.alphas[i].to(device)
            self.betas[i] = self.betas[i].to(device)

        self.params = torch.nn.ParameterList(self.W + self.b + self.alphas + self.betas)
        self.optimizer = self.optimizer_constructor(self.params)

    # non-linear activation function
    def activation(self, x, alpha, beta):

        return torch.multiply(torch.add(beta, torch.multiply(torch.sigmoid(torch.multiply(alpha, x)), torch.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict magnitudes
    def forward(self, parameters):

        output = torch.divide(torch.subtract(parameters, self.parameters_shift), self.parameters_scale)
        for i in range(self.n_layers - 1):

            # non-linear activation function
            output = self.activation(torch.add(torch.matmul(output, self.W[i]), self.b[i]), self.alphas[i], self.betas[i])

        # linear output layer
        output = torch.add(torch.matmul(output, self.W[-1]), self.b[-1])

        # rescale the output
        output = torch.add(torch.multiply(output, self.pca_scale), self.pca_shift)

        return output

    # save the state dict
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    # pass inputs through the network to predict spectrum
    def log_spectrum(self, parameters):

        # pass through network to compute PCA coefficients
        pca_coefficients = self.forward(parameters)

        # transform from PCA to normalized spectrum basis; shift and re-scale normalized spectrum -> spectrum
        return torch.add(torch.multiply(torch.matmul(pca_coefficients, self.pca_transform_matrix), self.log_spectrum_scale), self.log_spectrum_shift)

    ### Infrastructure for network training ###

    def compute_loss_spectra(self, spectra, parameters, noise_floor):

        return torch.sqrt(torch.mean(torch.divide(torch.square(torch.subtract( torch.exp(self.log_spectrum(parameters)), spectra)), torch.square(noise_floor))))

    def compute_loss_pca(self, pca, parameters):

      return torch.sqrt(torch.mean(torch.square(torch.subtract(self.forward(parameters), pca))))

    def compute_loss_log_spectra(self, log_spectra, parameters):

      return torch.sqrt(torch.mean(torch.square(torch.subtract(self.log_spectrum(parameters), log_spectra))))

    def training_step(self, theta, outputs, maxbatch=10000, loss_type='pca', noise_floor=None):

        if theta.shape[0] < maxbatch:

            # loss
            if loss_type == 'pca':
                loss = self.compute_loss_pca(theta, outputs)
            elif loss_type == 'log_spectra':
                loss = self.compute_loss_log_spectra(theta, outputs)
            elif loss_type =='spectra':
                loss = self.compute_loss_spectra(theta, outputs, noise_floor)

            # backprop
            loss.backward()

            # update
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss

        else:

            # create iterable dataset
            dataloader = DataLoader(TensorDataset(theta, outputs), batch_size=maxbatch)

            # loop over sub batches
            for theta_, outputs_ in dataloader:
                with torch.set_grad_enabled(True):

                    # loss
                    if loss_type == 'pca':
                        loss = self.compute_loss_pca(theta_, outputs_) * theta_.shape[0] / theta.shape[0]
                    elif loss_type == 'log_spectra':
                        loss = self.compute_loss_log_spectra(theta_, outputs_) * theta_.shape[0] / theta.shape[0]
                    elif loss_type =='spectra':
                        loss = self.compute_loss_spectra(theta_, outputs_, noise_floor)

                    # backprop
                    loss.backward()

            # update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss


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


class Photulator(torch.nn.Module):
    """
    PHOTULATOR model
    """

    def __init__(self, n_parameters=None, filters=None, parameters_shift=None, parameters_scale=None, magnitudes_shift=None, magnitudes_scale=None, n_hidden=[50,50], optimizer=lambda x: torch.optim.Adam(x, lr=1e-3), device='cpu'):

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

        # parameters
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.filters = filters
        self.n_filters = len(filters)

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_filters]
        self.n_layers = len(self.architecture) - 1

        # shifts and scales...

        # shifts and scales and transform matrix into tensorflow constants...

        # input parameters shift and scale
        self.parameters_shift = torch.tensor(parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters), dtype=torch.float32).to(device)
        self.parameters_scale = torch.tensor(parameters_scale if parameters_scale is not None else np.ones(self.n_parameters), dtype=torch.float32).to(device)

        # spectrum shift and scale
        self.magnitudes_shift = torch.tensor(magnitudes_shift if magnitudes_shift is not None else np.zeros(self.n_filters), dtype=torch.float32).to(device)
        self.magnitudes_scale = torch.tensor(magnitudes_scale if magnitudes_scale is not None else np.ones(self.n_filters), dtype=torch.float32).to(device)

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        for i in range(self.n_layers):
            self.W.append(torch.nn.Parameter( torch.sqrt(torch.tensor(2. / self.n_parameters)) * torch.randn((self.architecture[i], self.architecture[i+1])) ).to(device) )
            self.b.append(torch.nn.Parameter( torch.zeros((self.architecture[i+1]))).to(device))
        for i in range(self.n_layers-1):
            self.alphas.append(torch.nn.Parameter(torch.randn((self.architecture[i+1]))).to(device))
            self.betas.append(torch.nn.Parameter(torch.randn((self.architecture[i+1]))).to(device))

        # optimizer
        self.params = torch.nn.ParameterList(self.W + self.b + self.alphas + self.betas)
        #self.optimizer_constructor = optimizer
        #self.optimizer = self.optimizer_constructor(self.params)
        self.optimizer = optimizer(self.params)

    # change the device we're on
    def set_device(self, device):

        self.parameters_shift = self.parameters_shift.to(device)
        self.parameters_scale = self.parameters_scale.to(device)

        self.magnitudes_shift = self.magnitudes_shift.to(device)
        self.magnitudes_scale = self.magnitudes_scale.to(device)

        for i in range(self.n_layers):
            self.W[i] = self.W[i].to(device)
            self.b[i] = self.b[i].to(device)
        for i in range(self.n_layers-1):
            self.alphas[i] = self.alphas[i].to(device)
            self.betas[i] = self.betas[i].to(device)

        self.params = torch.nn.ParameterList(self.W + self.b + self.alphas + self.betas)
        #self.optimizer = self.optimizer_constructor(self.params)


    # non-linear activation function
    def activation(self, x, alpha, beta):

        return torch.multiply(torch.add(beta, torch.multiply(torch.sigmoid(torch.multiply(alpha, x)), torch.subtract(1.0, beta)) ), x)

    # call: forward pass through the network to predict magnitudes
    def forward(self, parameters):

        output = torch.divide(torch.subtract(parameters, self.parameters_shift), self.parameters_scale)
        for i in range(self.n_layers - 1):

            # non-linear activation function
            output = self.activation(torch.add(torch.matmul(output, self.W[i]), self.b[i]), self.alphas[i], self.betas[i])

        # linear output layer
        output = torch.add(torch.matmul(output, self.W[-1]), self.b[-1])

        # rescale the output
        output = torch.add(torch.multiply(output, self.magnitudes_scale), self.magnitudes_shift)

        return output

    # pass inputs through the network to predict spectrum
    def magnitudes(self, parameters):

        # call forward pass through network
        return self.forward(parameters)

    ### Infrastructure for network training ###

    def compute_loss(self, theta, mags):

        return torch.sqrt(torch.mean( torch.square(torch.subtract(self.forward(theta), mags)) ))


    def training_step(self, theta, mags, maxbatch=10000):

    	if theta.shape[0] < maxbatch:

    		# loss
	        loss = self.loss(theta, mags)

	        # backprop
	        loss.backward()

	        # update
	        self.optimizer.step()
	        self.optimizer.zero_grad()

	        return loss

    	else:

	    	# create iterable dataset
        	dataloader = DataLoader(TensorDataset(theta, mags), batch_size=maxbatch)

	        # loop over sub batches
	        for theta_, mags_ in dataloader:
	            with torch.set_grad_enabled(True):

	                # loss
	                loss = self.loss(theta_, mags_) * theta_.shape[0] / theta.shape[0]

	                # backprop
	                loss.backward()

	        # update parameters
	        self.optimizer.step()
	        self.optimizer.zero_grad()

	        return loss


class PhotulatorModelStack:

    def __init__(self, root_dir, filenames, device="cpu"):

        # how many emulators?
        self.n_emulators = len(filenames)

        # load emulator models
        self.emulators = [torch.load(filename) for filename in filenames]

        # change device if neccessary
        for i in range(self.n_emulators):
            self.emulators[i].set_device(device)

        # log10 constant
        self.ln10 = torch.tensor(np.log(10.), dtype=torch.float32).to(device)

    # compute fluxes (in units of nano maggies) given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def fluxes(self, theta, N):

        return torch.concat([torch.exp( torch.multiply(torch.add(torch.multiply(-0.4, torch.add(self.emulators[i].forward(theta), torch.unsqueeze(N, -1))), 9.), self.ln10) ) for i in range(self.n_emulators)], axis=-1)

    # compute magnitudes given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def magnitudes(self, theta, N):

        return torch.concat([torch.add(self.emulators[i].forward(theta), torch.unsqueeze(N, -1)) for i in range(self.n_emulators)], axis=-1)


# train photulator model stack
def train_photulator_stack(training_theta, training_mag, parameters_shift, parameters_scale, magnitudes_shift, magnitudes_scale, n_layers=4, n_units=128, filters=None, validation_split=0.1, lr=[1e-3, 1e-4, 1e-5, 1e-6], batch_size=[1000, 10000, 50000, 1000000], maxbatch=10000, epochs=1000, patience=20, root_dir='', verbose=True, device='cpu', optimizer=lambda x: torch.optim.Adam(x, lr=1e-3), all_on_device=False):

    # put the training data all on the device if we want it there
    if all_on_device:
        training_theta = training_theta.to(device)
        training_mag = training_mag.to(device)

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
                           n_hidden=[n_units]*n_layers,
                           optimizer=optimizer,
                           device=device)

        # train using cooling/heating schedule for lr/batch-size
        for i in range(len(lr)):

            if verbose is True:
                print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]))

            # set learning rate
            optimizer_state_dict = photulator.optimizer.state_dict()
            optimizer_state_dict['param_groups'][0]['lr'] = lr[i]
            photulator.optimizer.load_state_dict(optimizer_state_dict)

            # dataset and dataloader
            dataset = TensorDataset(training_theta, torch.unsqueeze(training_mag[:,f],-1))
            training_data, validation_data = torch.utils.data.random_split(dataset, [int(len(dataset)*(1.-validation_split)), len(dataset) - int(len(dataset)*(1.-validation_split))])
            training_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)

            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            best_state = photulator.state_dict()
            patience_counter = 0

            # loop over epochs
            while patience_counter < patience:

                # loop over batches for a single epoch
                for theta, mag in training_dataloader:

                    # move to correct device
                    theta.to(device)
                    mag.to(device)

                    # training step
                    loss = photulator.training_step(theta, mag, maxbatch=maxbatch)

                # compute total loss and validation loss
                validation_theta, validation_mag = validation_data[:]
                validation_loss.append(photulator.compute_loss(validation_theta, validation_mag).cpu().detach().numpy())

                # early stopping condition
                if validation_loss[-1] < best_loss:
                    best_loss = validation_loss[-1]
                    best_state = photulator.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    photulator.load_state_dict(best_state)
                    torch.save(photulator, root_dir + 'model_{}x{}_'.format(n_layers, n_units) + filters[f] + '.pt')
                    if verbose is True:
                        print('Validation loss = ' + str(best_loss))
                    break

        # save CPU version of the model by default
        photulator.set_device('cpu')
        torch.save(photulator, root_dir + 'model_{}x{}_'.format(n_layers, n_units) + filters[f] + '.pt')
