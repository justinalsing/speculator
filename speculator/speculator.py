import numpy as np
import torch
import pickle
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
from torch.distributions.transforms import StackTransform, identity_transform
import wandb

class SqrtTransform(torch.nn.Module):
    def forward(self, x):
        return torch.sqrt(x)

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
            self.load_state_dict(torch.load(restore_filename, map_location=device))

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

    def __init__(self, n_parameters=None, 
        filters=None, 
        parameters_shift=None, 
        parameters_scale=None, 
        magnitudes_shift=None, 
        magnitudes_scale=None, 
        f_b=None, 
        n_hidden=[50,50], 
        sigma_init=1e-3,
        parameter_names=None):

        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param filters: list of filter names
        :param parameters_shift: shift for input parameters
        :param parameters_scale: scale for input parameters
        :param magnitudes_shift: shift for the output mags
        :param magnitudes_scale: scale for the output mags
        :param n_hidden: list with number of hidden units for each hidden layer
        :param sigma_init: std dev of weight and bias initization
        :param transform: StackedTransform for transforming parameters before passing to network
        :param parameter_names: list of the names of the parameters that the model expects as inputs when calling it
        """

        # super
        super(Photulator, self).__init__()

        # parameters
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.filters = filters
        self.n_filters = len(filters)
        self.parameter_names = parameter_names

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_filters]
        self.n_layers = len(self.architecture) - 1

        # shifts and scales...

        # shifts and scales and transform matrix into tensorflow constants...

        # input parameters shift and scale
        self.register_buffer('parameters_shift', torch.tensor(parameters_shift if parameters_shift is not None else np.zeros(self.n_parameters), dtype=torch.float32))
        self.register_buffer('parameters_scale', torch.tensor(parameters_scale if parameters_scale is not None else np.ones(self.n_parameters), dtype=torch.float32))

        # spectrum shift and scale
        self.register_buffer('magnitudes_shift', torch.tensor(magnitudes_shift if magnitudes_shift is not None else np.zeros(self.n_filters), dtype=torch.float32))
        self.register_buffer('magnitudes_scale', torch.tensor(magnitudes_scale if magnitudes_scale is not None else np.ones(self.n_filters), dtype=torch.float32))

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = torch.nn.ParameterList( [ torch.nn.Parameter( sigma_init * torch.randn((self.architecture[i], self.architecture[i+1])) ) for i in range(self.n_layers)] )
        self.b = torch.nn.ParameterList( [ torch.nn.Parameter( sigma_init * torch.randn((self.architecture[i+1])) ) for i in range(self.n_layers)] )
        self.alphas = torch.nn.ParameterList( [ torch.nn.Parameter( sigma_init * torch.randn((self.architecture[i+1])) ) for i in range(self.n_layers)] )
        self.betas = torch.nn.ParameterList( [ torch.nn.Parameter(sigma_init * torch.randn((self.architecture[i+1]))) for i in range(self.n_layers)] )

        # luptitude parameters
        self.register_buffer('f_b', torch.tensor(0., dtype=torch.float32) if f_b is None else torch.tensor(f_b, dtype=torch.float32) )
        self.register_buffer('ln10', torch.tensor(np.log(10), dtype=torch.float32) )

    # non-linear activation function
    @torch.jit.export
    def activation(self, x, alpha, beta):

        return torch.multiply(torch.add(beta, torch.multiply(torch.sigmoid(torch.multiply(alpha, x)), torch.subtract(torch.tensor(1.0, dtype=torch.float32, device=beta.device), beta)) ), x)

    # call: forward pass through the network to predict magnitudes
    # by default this should predict absolute unit mass magnitudes, in units of nano-maggies
    def forward(self, parameters):

        # shift and scale
        output = torch.divide(torch.subtract(parameters, self.parameters_shift), self.parameters_scale)

        # layers
        #for i in range(self.n_layers - 1):

            # non-linear activation function
        #    output = self.activation(torch.add(torch.matmul(output, self.W[i]), self.b[i]), self.alphas[i], self.betas[i])

        # linear output layer
        #output = torch.add(torch.matmul(output, self.W[-1]), self.b[-1])

        # layers
        for i, (W, b, alpha, beta) in enumerate(zip(self.W, self.b, self.alphas, self.betas)):

            # non-linear activation function
            output = self.activation(torch.add(torch.matmul(output, W), b), alpha, beta)

        # rescale the output
        output = torch.add(torch.multiply(output, self.magnitudes_scale), self.magnitudes_shift)

        return output

    # compute fluxes in maggies
    @torch.jit.export
    def flux(self, parameters, N):

        return torch.exp( torch.multiply(torch.multiply(torch.tensor(-0.4, dtype=torch.float32, device=parameters.device), self.magnitudes(parameters, N)), self.ln10) )

    # pass inputs through the network to predict apparent magnitudes (in standard magnitude units)
    @torch.jit.export
    def magnitudes(self, parameters, N):

        return torch.add(self.forward(parameters), N)

    # pass inputs through the network to predict asinh magnitudes (in standard magnitude units)
    def luptitudes(self, parameters, N):

        # absolute magnitudes -> flux in nano maggies
        flux = torch.multiply(self.flux(parameters, N), torch.tensor(1e9, dtype=torch.float32, device=parameters.device))

        # flux in nano maggies -> luptitudes (in mormal magnitude units)
        return flux2asinhmag(flux, self.f_b)

    ### Infrastructure for network training ###

    def compute_loss_absolute_magnitudes(self, theta, N, mags):

        return torch.sqrt(torch.mean( torch.square(torch.subtract(self.forward(theta), mags)) ))

    def compute_loss_luptitudes(self, theta, N, mags):

        return torch.sqrt(torch.mean( torch.square(torch.subtract(self.luptitudes(theta, N), mags)) ))


    def training_step_absolute_magnitudes(self, theta, N, mags, optimizer):

        # zero the gradients first
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            # loss
            loss = self.compute_loss_absolute_magnitudes(theta, N, mags)

            # backprop
            loss.backward()

        # update
        optimizer.step()

        return loss


    def training_step_absolute_magnitudes_accumulated(self, theta, N, mags, optimizer, maxbatch=10000):

        # zero the gradients first
        optimizer.zero_grad()

        # create iterable dataset
        dataloader = DataLoader(TensorDataset(theta, N, mags), batch_size=maxbatch)

        # loop over sub batches
        for theta_, N_, mags_ in dataloader:
            with torch.set_grad_enabled(True):

                # loss
                loss = self.compute_loss_absolute_magnitudes(theta_, N_, mags_) * torch.true_divide(theta_.shape[0], theta.shape[0])

                # backprop
                loss.backward()

        # update parameters
        optimizer.step()

        return loss

    def training_step_luptitudes(self, theta, N, mags, optimizer):

        # zero the gradients first
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            # loss
            loss = self.compute_loss_luptitudes(theta, N, mags)

            # backprop
            loss.backward()

        # update
        optimizer.step()

        return loss


    def training_step_luptitudes_accumulated(self, theta, N, mags, optimizer, maxbatch=10000):

        # zero the gradients first
        optimizer.zero_grad()

        # create iterable dataset
        dataloader = DataLoader(TensorDataset(theta, N, mags), batch_size=maxbatch)

        # loop over sub batches
        for theta_, N_, mags_ in dataloader:
            with torch.set_grad_enabled(True):

                # loss
                loss = self.compute_loss_luptitudes(theta_, N_, mags_) * torch.true_divide(theta_.shape[0], theta.shape[0])

                # backprop
                loss.backward()

        # update parameters
        optimizer.step()

        return loss

class PhotulatorModelStack:

    def __init__(self, root_dir, filenames, device="cpu"):

        # how many emulators?
        self.n_emulators = len(filenames)

        # load emulator models
        self.emulators = [torch.load(filename).to(device) for filename in filenames]

    # compute fluxes (in units of nano maggies) given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def fluxes(self, theta, N):

        return torch.concat([self.emulators[i].fluxes(theta, N) for i in range(self.n_emulators)], axis=-1)

    # compute magnitudes given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def magnitudes(self, theta, N):

        return torch.concat([self.emulators[i].magnitudes(theta, N) for i in range(self.n_emulators)], axis=-1)

    # compute magnitudes given SPS parameters (theta) and normalization (N = -2.5log10M + dm(z))
    def luptitudes(self, theta, N):

        return torch.concat([self.emulators[i].luptitudes(theta, N) for i in range(self.n_emulators)], axis=-1)

# train photulator model stack
def train_photulator_stack(training_theta, training_N, training_mag, parameters_shift, parameters_scale, magnitudes_shift, magnitudes_scale, parameter_names=None, n_layers=4, n_units=128, filters=None, validation_split=0.1, lr=[1e-3, 1e-4, 1e-5, 1e-6], batch_size=[1000, 10000, 50000, 1000000], maxbatch=100000, maxepochs=500, patience=20, root_dir='', verbose=True, device='cuda', all_on_device=True, wandb_init=None, loss_in='absmag', f_b=None, sigma_init=1e-2):

    # put the training data all on the device if we want it there
    if all_on_device is True:
        training_theta = training_theta.to(device)
        training_N = training_N.to(device)
        training_mag = training_mag.to(device)

    # architecture
    n_hidden = [n_units]*n_layers

    # how many training rounds to do?
    rounds = len(lr)

    # train each band in turn
    for f in range(len(filters)):

        if wandb_init is not None:
            wandb.init(name=wandb_init['name'] + '_' + filters[f], project=wandb_init['project'])

        if verbose is True:
            print('filter ' + filters[f] + '...')

        # construct the PHOTULATOR model
        photulator = torch.jit.script(Photulator(n_parameters=training_theta.shape[-1],
                           filters=[filters[f]],
                           parameters_shift=parameters_shift,
                           parameters_scale=parameters_scale,
                           magnitudes_shift=magnitudes_shift[f],
                           magnitudes_scale=magnitudes_scale[f],
                           n_hidden=[n_units]*n_layers,
                           f_b=f_b[f],
                           sigma_init=sigma_init,
                           parameter_names=parameter_names)).to(device)

        # construct an optimizer
        optimizer = torch.optim.Adam(photulator.parameters())

        # train using cooling/heating schedule for lr/batch-size
        for i in range(rounds):

            if verbose is True:
                print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]))

            # set learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr[i]

            # dataset and dataloader
            dataset = TensorDataset(training_theta, training_N, torch.unsqueeze(training_mag[:,f],-1))
            training_data, validation_data = torch.utils.data.random_split(dataset, [int(len(dataset)*(1.-validation_split)), len(dataset) - int(len(dataset)*(1.-validation_split))])
            validation_theta, validation_N, validation_mag = validation_data[:]
            training_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size[i], num_workers=4, pin_memory=True)
            epochs_per_step = 1. / len(training_dataloader)
            epoch = 0.
            
            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            best_state = photulator.state_dict()
            patience_counter = 0

            # which training step to use?
            if loss_in == 'absmag':
                compute_loss = lambda theta, N, mag: torch.sqrt(torch.mean( torch.square(torch.subtract(photulator.forward(theta), mag)) ))
            elif loss_in == 'asinhmag':
                compute_loss = lambda theta, N, mag: torch.sqrt(torch.mean( torch.square(torch.subtract(flux2asinhmag(photulator.flux(theta, N) * 1e9, photulator.f_b), mag)) ))

            # loop over epochs
            while patience_counter < patience and epoch < maxepochs:

                # loop over batches for a single epoch
                for theta, N, mag in training_dataloader:

                    # training step..

                    # zero gradients
                    optimizer.zero_grad()

                    # backprop and step
                    if theta.shape[0] < maxbatch:
                        loss = compute_loss(theta.to(device, non_blocking=True), N.to(device, non_blocking=True), mag.to(device, non_blocking=True))
                        loss.backward()
                        optimizer.step()
                    else:
                        # create iterable dataset
                        minidataloader = DataLoader(TensorDataset(theta, N, mag), batch_size=maxbatch)

                        # loop over sub batches
                        for theta_, N_, mags_ in minidataloader:
                            with torch.set_grad_enabled(True):

                                # loss
                                loss = compute_loss(theta_.to(device, non_blocking=True), N_.to(device, non_blocking=True), mags_.to(device, non_blocking=True)) * torch.true_divide(theta_.shape[0], theta.shape[0])

                                # backprop
                                loss.backward()

                        # update parameters
                        optimizer.step()

                    #loss = training_step(theta.to(device, non_blocking=True), N.to(device, non_blocking=True), mag.to(device, non_blocking=True), optimizer)

                    # increment epoch
                    epoch += epochs_per_step

                    # update wandb if needed
                    if wandb_init is not None:
                        wandb.log({'train_loss':loss.detach().cpu().item(), 'epoch':epoch})

                # compute total loss and validation loss
                validation_loss.append(compute_loss(validation_theta.to(device), validation_N.to(device), validation_mag.to(device)).cpu().detach().numpy())

                # early stopping condition
                if validation_loss[-1] < best_loss:
                    best_loss = validation_loss[-1]
                    best_state = photulator.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    photulator.load_state_dict(best_state)
                    photulator.save(root_dir + 'model_{}x{}_'.format(n_layers, n_units) + filters[f] + '.pt')
                    torch.save(best_state, root_dir + 'model_{}x{}_state'.format(n_layers, n_units) + filters[f] + '.pt')
                    if verbose is True:
                        print('Validation loss = ' + str(best_loss))
                    break

                # update wandb if needed
                if wandb_init is not None:
                    wandb.log({'val_loss':validation_loss[-1], 'best_loss':best_loss, 'patience_counter':patience_counter, 'epoch':epoch})

        if wandb_init is not None:
            wandb.finish()

        # save CPU version of the model by default
        photulator.to('cpu')
        photulator.save(root_dir + 'model_{}x{}_'.format(n_layers, n_units) + filters[f] + '.pt')
        torch.save(photulator.state_dict(), root_dir + 'model_{}x{}_state'.format(n_layers, n_units) + filters[f] + '.pt')

# magnitude conversion functions

# flux in nano maggies to apparent magnitudes
def flux2mag(flux):
    return -2.5 * torch.log10(flux) + 22.5

# flux in nano maggies to asinh magnitudes
def flux2asinhmag(flux, f_b):

    """
    Computes the asinh magnitudes from fluxes

    flux: torch tensor, should be in units of nano maggies
    f_b: flux below which the asinh magnitude is linear, should be in units of nano maggies

    """

    asinh_mag = -1.0857362047581294 * (torch.arcsinh(flux/(2.0 * f_b)) - torch.log(10**9 / f_b))

    return asinh_mag

# asinh magnitudes to fluxes in nano maggies
def asinhmag2flux(asinh_mag, f_b):

    """
    Computes fluxes in nano maggies from asinh magnitudes

    asinh_magnitudes: torch tensor, should be in normal magnitude units
    f_b: flux below which the asinh magnitude is linear, should be in units of nanomaggies
    f_0: reference flux, default is 1 jansky or 10^9 nanomaggies

    """

    return torch.sinh(-(asinh_mag / -1.0857362047581294) + torch.log(10**9 / f_b) ) * 2 * f_b 

# magnitudes to asinh magnitudes
def mag2asinhmag(mag, f_b):
    return flux2asinhmag(10**(-0.4 * (mag - 22.5)), f_b)

# asinh magnitudes to magnitudes
def asinhmag2mag(asinhmag, f_b):

    return flux2mag( torch.sinh( asinhmag / (-1.0857362047581294) + torch.log(10**9 / f_b) ) * 2.0 * f_b )


