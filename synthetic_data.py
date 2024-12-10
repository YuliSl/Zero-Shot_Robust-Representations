import numpy as np

def sample_means(v0, v0_dim, vplus, vplus_dim, vminus, vminus_dim, noise_dim, n1, n2):
  """Samples means from a multivariate normal distribution centered around 0 with diagonal covariance (described in Section 2).
    """
  signal_dim = v0_dim + vminus_dim + vplus_dim
  total_dim = signal_dim + noise_dim

  Mu = np.zeros(total_dim)

  diag_1 = np.hstack([np.repeat(v0, v0_dim), np.repeat(vplus, vplus_dim), np.repeat(vminus, vminus_dim), np.repeat(0.0, noise_dim)])
  Sigma1 = np.eye(total_dim)*diag_1

  diag_2 = np.hstack([np.repeat(v0, v0_dim), np.repeat(vminus, vminus_dim), np.repeat(vplus, vplus_dim), np.repeat(0.0, noise_dim)])
  Sigma2 = np.eye(total_dim)*diag_2

  return np.vstack([np.random.multivariate_normal(Mu, Sigma1, size=n1), np.random.multivariate_normal(Mu, Sigma2, size=n2)])


def shuffle(z, c):
  """Shuffles data points and labels
  """
  idx = np.random.permutation(len(z))
  return z[idx], c[idx]


def sample_Z(vz, signal_dim, vz_noise, noise_dim, means, r, c_range):
  """Samples r data points for each class from a multivariate normal distribution centered around the class mean with diagonal covariance. (described in Section 2).
  """
  total_dim = signal_dim + noise_dim
  z_Sigma = np.hstack([np.repeat(vz, signal_dim), np.repeat(vz_noise, noise_dim)])*np.eye(total_dim)
  z = np.vstack([np.random.multivariate_normal(m, z_Sigma, size=r) for m in means])
  c = np.hstack([np.repeat(i, r) for i in c_range])
  z, c = shuffle(z, c)
  return z, c


def generate_synthetic_data(Nc, r, v0, vminus, vplus, vz, vz_noise, v0_dim, vminus_dim, vplus_dim, noise_dim, p_minor):
  """Generates training, validation (in distribution), and test (distribution-shift) data
   """
  p_major = 1 - p_minor
  n_minor = int(p_minor*Nc)
  n_major = Nc - n_minor

  signal_dim = v0_dim + vminus_dim + vplus_dim
  total_dim = signal_dim + noise_dim

  # train data
  means_train = sample_means(v0, v0_dim, vplus, vplus_dim, vminus, vminus_dim, noise_dim, n_major, n_minor)
  z_train, c_train  = sample_Z(vz, signal_dim, vz_noise, noise_dim, means_train, r, range(0, Nc))

  # validation data
  means_val = sample_means(v0, v0_dim, vplus, vplus_dim, vminus, vminus_dim, noise_dim, n_major, n_minor)
  z_val, c_val = sample_Z(vz, signal_dim, vz_noise, noise_dim, means_val, r,range(0, Nc)) 

  # test data
  means_test = sample_means(v0, v0_dim, vplus, vplus_dim, vminus, vminus_dim, noise_dim, n_minor, n_major)
  z_test, c_test = sample_Z(vz, signal_dim, vz_noise, noise_dim, means_test, r, range(Nc, 2*Nc))

  return z_train, c_train, z_val, c_val, z_test, c_test 
