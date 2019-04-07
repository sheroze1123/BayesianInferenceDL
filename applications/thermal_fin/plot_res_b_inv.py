import numpy as np
#  from scipy.stats import norm.pdf as gau
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')

def gau(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def plot_norm(mu_dl, mu_rom, mu_fom, var_dl, var_rom, var_fom, true_mu, idx):
  sigma_dl = np.sqrt(var_dl)
  sigma_rom = np.sqrt(var_rom)
  sigma_fom = np.sqrt(var_fom)
  x = np.linspace(0.1, 1, 300)
  plt.cla()
  plt.clf()
  plt.plot(x, gau(x, mu_dl, sigma_dl), color='b')
  plt.plot(x, gau(x, mu_rom, sigma_rom), color='g')
  plt.plot(x, gau(x, mu_fom, sigma_fom), color='k')
  plt.axvline(x=true_mu,color='r')
  plt.legend(["rom + deep learning", "rom", "full order model", "true"])
  plt.xlabel("Thermal conductivity of subfin")
  plt.title("Bayesian inference of thermal conductivity of subfin {}".format(idx))
  plt.savefig("plots/b_inv_fin_{}.png".format(idx), dpi=250) 
  #  plt.show()

z_dl = np.array([0.38157034, 0.58727851, 0.70760877, 0.80090808, 0.25971255])
z_true = np.array([0.41126864, 0.61789679, 0.75873243, 0.96527541, 0.223480755])
cov_dl = np.array([0.00017591, 0.001135388, 0.00341071, 0.01834024, 0.00039539])
z_fom = np.array([0.39791155, 0.58295775, 0.71561434, 0.84577546, 0.24091571])
z_rom = np.array([0.35279272, 0.27739859, 0.59216552, 0.45366806, 0.29029221])
cov_rom = np.array([4.88774011e-05, 1.78930633e-04, 8.80371721e-05, 4.41085694e-05, 2.54974576e-05])
cov_fom = np.array([0.00046767, 0.00260971, 0.00488985, 0.03551424, 0.00079161])

for i in range(5):
  plot_norm(z_dl[i], z_rom[i], z_fom[i], cov_dl[i], cov_rom[i], cov_fom[i], z_true[i], i) 
