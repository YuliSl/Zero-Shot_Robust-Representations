import numpy as np
import tensorflow as tf

def contrastive_loss(y_true, y_pred, margin=1.0): 
    """Computes the contrastive loss between `y_true` and `y_pred`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0.0))

def distinct_pairs_func(v):
  """ Computes all distinct pairs
  """
  all_pairs = np.array(np.meshgrid(v, v)).T.reshape(-1, 2)
  distinct_pairs = np.unique(all_pairs, axis=0)
  return distinct_pairs[distinct_pairs[:,0] != distinct_pairs[:,1]]

def make_pairs(Z, C, pos_per_class=1000):
  """ Samples positive and negative pairs
  """
  unq_c = np.unique(C)
  Z1, Z2, Y, Cs = [], [], [], []
  for c in unq_c:
    same_class_idx = np.where(C==c)[0]
    disticnt_same_class_pairs = distinct_pairs_func(same_class_idx)
    np.random.shuffle(disticnt_same_class_pairs)

    n = len(disticnt_same_class_pairs)
    if pos_per_class < n:
      n = pos_per_class

    # positive pairs
    Z1.append(Z[disticnt_same_class_pairs[0:n,0]])
    Z2.append(Z[disticnt_same_class_pairs[0:n,1]])
    Y.append(np.ones(n))

    # negative pairs
    np.random.shuffle(disticnt_same_class_pairs)
    Z1.append(Z[disticnt_same_class_pairs[0:n,0]])
    diff_class_idx = np.where(C!=c)[0]
    rnd_diff_idx = np.random.choice(diff_class_idx, n)
    Z2.append(Z[rnd_diff_idx])
    Y.append(np.zeros(n))

    Cs.append(np.vstack([np.repeat(c, 2*n) ,np.hstack([np.repeat(c, n), C[rnd_diff_idx]])]))

  Z1, Z2, Y, Cs = np.vstack(Z1), np.vstack(Z2), np.hstack(Y), np.hstack(Cs).T
  return Z1, Z2, Y, Cs
