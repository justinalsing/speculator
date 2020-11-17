import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import matplotlib as mpl
import numpy as np
import scipy.stats as stats

def triangle_plot(samples = None, weights = None, truths = None, savefig = False, filename = None, names = None, labels = None, ranges = None, fontsize = 14, legend_labels=None):

    # Set samples to the posterior samples by default
    if weights is None:
        mc_samples = [MCSamples(samples=samples[i], weights=None, names=names, labels=labels, ranges=ranges) for i in range(len(samples))]
    else:
        mc_samples = [MCSamples(samples=samples[i], weights=weights[i], names=names, labels=labels, ranges=ranges) for i in range(len(samples))]

    # Triangle plot
    plt.close()
    with mpl.rc_context():
        g = plots.getSubplotPlotter(width_inch = 12)
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.6
        g.settings.axes_fontsize=fontsize
        g.settings.legend_fontsize=fontsize
        g.settings.lab_fontsize=fontsize
        g.triangle_plot(mc_samples, filled_compare=True, normalized=True, legend_labels=legend_labels)
        for i in range(0, len(samples[0][0,:])):
            for j in range(0, i+1):
                ax = g.subplots[i,j]
                #xtl = ax.get_xticklabels()
                #ax.set_xticklabels(xtl, rotation=45)
        if truths is not None:
            for column in range(0, len(samples[0][0,:])-1):
                for row in range(column+1, len(samples[0][0,:])): 
                    ax = g.subplots[row,column]
                    for t in range(len(truths)):
                        ax.scatter(np.array([truths[t][column]]), np.array([truths[t][row]]), marker = 'x', color = 'black')
        #plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)

        if savefig:
            plt.savefig(filename)
            plt.show()
        else:
            plt.show()

        plt.close()

# plot posterior fit to photometry in data-space
def plot_data_space_fit(fluxes_observed, fluxes_sigma, posterior_fluxes, obs, savefig=False, filename=None):
    
    plt.close()
    columnwidth = 25 # cm
    aspect = 1.67
    pts_per_inch = 72.27
    inch_per_cm = 2.54
    width = columnwidth/inch_per_cm
    plt.rcParams.update({'figure.figsize': [width, width / aspect],
                                    'backend': 'pdf',
                                    'font.size': 14,
                                    'legend.fontsize': 14,
                                    'legend.frameon': False,
                                    'legend.loc': 'best',
                                    'lines.markersize': 3,
                                    'lines.linewidth': 2,
                                    'axes.linewidth': .5,
                                    'axes.edgecolor': 'black'})

    # effective wavelengths for each band
    wavelengths = [obs['filters'][i].wave_effective for i in range(9)]

    # plot data vs fitted fluxes
    for i in range(9):
        if i == 0:
            plt.errorbar([wavelengths[i]], [fluxes_observed[i]], yerr=[fluxes_sigma[i]], fmt=".k", elinewidth=10, label='obs $\pm 1\sigma$')
        else:
            plt.errorbar([wavelengths[i]], [fluxes_observed[i]], yerr=[fluxes_sigma[i]], fmt=".k", elinewidth=10)
        violin = plt.violinplot(posterior_fluxes[:,i], positions=[wavelengths[i]], widths=[200.])

        violin['bodies'][0].set_color('red')
        violin['cbars'].set_color('red')
        violin['cmaxes'].set_color('red')
        violin['cmins'].set_color('red')
        
    # fake stuff for legend   
    plt.plot([-1000, -1001], [-1000, -1001], lw = 3, color = 'red', label = 'posterior samples', alpha = 0.5)
    plt.legend()
    plt.xlim(2000, 23000)
    plt.yscale('log')
    plt.xlabel('wavelength [$\AA$]')
    plt.ylabel('flux [maggies]')
    
    if savefig:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()
    plt.close()
        
# plot posterior fit to fluxes per band on linear scales

def plot_data_space_residuals(fluxes_observed, fluxes_sigma, posterior_fluxes, savefig=False, filename=None):
    

    plt.close()
    columnwidth = 25 # cm
    aspect = 1.67
    pts_per_inch = 72.27
    inch_per_cm = 2.54
    width = columnwidth/inch_per_cm
    plt.rcParams.update({'figure.figsize': [width, width / aspect],
                                    'backend': 'pdf',
                                    'font.size': 14,
                                    'legend.fontsize': 14,
                                    'legend.frameon': False,
                                    'legend.loc': 'best',
                                    'lines.markersize': 3,
                                    'lines.linewidth': 2,
                                    'axes.linewidth': .5,
                                    'axes.edgecolor': 'black'})

    filters = ['u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']
    fig, ax = plt.subplots(3, 3)
    ax = np.ravel(ax)
    
    for band in range(9):

        ax[band].set_title(filters[band])
        ax[band].errorbar([-.1], [fluxes_observed[band]], yerr=[fluxes_sigma[band]], color='black', capsize=4)
        violin = ax[band].violinplot(posterior_fluxes[:,band], positions=[.1], widths=[0.1])
        violin['bodies'][0].set_color('red')
        violin['cbars'].set_color('red')
        violin['cmaxes'].set_color('red')
        violin['cmins'].set_color('red')
        
        ax[band].set_xlim(-0.3, 0.3)
        ax[band].xaxis.set_visible(False)
        
        dy = max( fluxes_sigma[band], max(max(posterior_fluxes[:,band])-fluxes_observed[band], abs(min(posterior_fluxes[:,band]) - fluxes_observed[band])))
        ax[band].set_ylim(fluxes_observed[band] - 1.2*dy, fluxes_observed[band] + 1.2*dy)
        ax[band].axhline(fluxes_observed[band], color = 'black')
        
        if band == 3:
            ax[band].set_ylabel('fluxes [maggies]')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()

    plt.close()
        
def plot_data_space_residuals_row(fluxes_observed, fluxes_sigma, posterior_fluxes, savefig=False, filename=None):
    

    plt.close()
    columnwidth = 35 # cm
    aspect = 4*1.67
    pts_per_inch = 72.27
    inch_per_cm = 2.54
    width = columnwidth/inch_per_cm
    plt.rcParams.update({'figure.figsize': [width, width / aspect],
                                    'backend': 'pdf',
                                    'font.size': 14,
                                    'legend.fontsize': 14,
                                    'legend.frameon': False,
                                    'legend.loc': 'best',
                                    'lines.markersize': 3,
                                    'lines.linewidth': 2,
                                    'axes.linewidth': .5,
                                    'axes.edgecolor': 'black'})

    filters = ['u', 'g', 'r', 'i', 'Z', 'Y', 'J', 'H', 'Ks']
    fig, ax = plt.subplots(1, 9)
    ax = np.ravel(ax)
    
    for band in range(9):

        ax[band].set_title(filters[band])
        violin = ax[band].violinplot(posterior_fluxes[:,band], positions=[0.], widths=[0.1])
        violin['bodies'][0].set_color('red')
        violin['cbars'].set_color('red')
        violin['cmaxes'].set_color('red')
        violin['cmins'].set_color('red')
        ax[band].errorbar([0.], [fluxes_observed[band]], yerr=[fluxes_sigma[band]], color='black', capsize=4)
        
        ax[band].set_xlim(-0.2, 0.2)
        ax[band].xaxis.set_visible(False)
        ax[band].yaxis.set_visible(False)

        dy = max( fluxes_sigma[band], max(max(posterior_fluxes[:,band])-fluxes_observed[band], abs(min(posterior_fluxes[:,band]) - fluxes_observed[band])))
        ax[band].set_ylim(fluxes_observed[band] - 1.2*dy, fluxes_observed[band] + 1.2*dy)
        ax[band].axhline(fluxes_observed[band], color = 'black')
        
        if band == 0:
            ax[band].set_ylabel('fluxes [maggies]')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    if savefig:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()

    plt.close()



def plot_redshift_marginal(z_samples, spec_z, bpz_z, savefig=False, filename=None):

    plt.close()
    columnwidth = 20 # cm
    aspect = 1.67
    pts_per_inch = 72.27
    inch_per_cm = 2.54
    width = columnwidth/inch_per_cm
    plt.rcParams.update({'figure.figsize': [width, width / aspect],
                                    'backend': 'pdf',
                                    'font.size': 14,
                                    'legend.fontsize': 14,
                                    'legend.frameon': False,
                                    'legend.loc': 'best',
                                    'lines.markersize': 3,
                                    'lines.linewidth': 2,
                                    'axes.linewidth': .5,
                                    'axes.edgecolor': 'black'})

    # histogram
    plt.hist(z_samples, bins = 50, alpha = 0.3, density=True, color = 'blue')
    
    # KDE
    x = np.linspace(min(z_samples), max(z_samples), 500)
    y = stats.gaussian_kde(z_samples)(x)
    plt.plot(x, y, color = 'blue', lw = 3, alpha = 0.5)
    
    # set x limits
    zmin = min(min(z_samples) - 0.015, spec_z - 0.015, bpz_z - 0.015)
    zmax = max(max(z_samples) + 0.015, spec_z + 0.015, bpz_z + 0.015)
    plt.xlim(zmin, zmax)
    
    # spec and BPZ estimates
    plt.axvline(spec_z, color = 'red', lw = 3, alpha = 0.5)
    plt.axvline(bpz_z, color = 'orange')
    plt.text(spec_z + 0.001, plt.gca().get_ylim()[-1]*0.5, 'spec-z = {:.2f}'.format(spec_z), rotation=90, color = 'red')
    plt.text(bpz_z + 0.001, plt.gca().get_ylim()[-1]*0.5, 'BPZ = {:.2f}'.format(bpz_z), rotation=90, color = 'orange')
    
    # polish
    plt.xlabel('redshift, $z$')
    plt.gca().yaxis.set_ticklabels([])
    plt.ylabel('posterior density')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(filename)
        plt.show()
    else:
        plt.show()

    plt.close()