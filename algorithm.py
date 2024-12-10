import numpy as np
import tensorflow as tf
from pairs import distinct_pairs_func, make_pairs
from sklearn.metrics import roc_curve, auc

from pairs import contrastive_loss

def cosine_distance(a, b):
  """Cosine distance, normalized to 0-1 range
  """
  sim = tf.reduce_sum(a * b, axis=1) / (tf.linalg.norm(a, axis=1) * tf.linalg.norm(b, axis=1))
  return (1-sim)/2


def compute_loss(z1_hat, z2_hat, y_true, margin=0.5):
  """Average cosine distance between embeddings
  """
  dist = cosine_distance(z1_hat, z2_hat)
  l = tf.reduce_mean(contrastive_loss(y_true, dist, margin))
  return l, dist


 # ~~~~~  IRM  ~~~~~ #

def IRM_penalty(z1_hat, z2_hat, y):
  """IRM penalty on a single enviroment
  """
  scale = tf.constant(1.)
  with tf.GradientTape() as local_tape:
    local_tape.watch(scale)
    local_loss = compute_loss(z1_hat*scale, z2_hat*scale, y, margin=0.5)
  grad = local_tape.gradient(local_loss, [scale])[0]
  return tf.reduce_sum(tf.math.pow(grad, 2.0))


def IRM_loss(weights, avg_loss, avg_penalty, step, l2_regularizer_weight=tf.constant(0.001), predefined_penalty_weight=10, penalty_anneal_iters=20):
  """IRM loss with scaled weights
  """
  weight_norm = tf.constant(0.0)
  for w in weights:
    weight_norm += tf.math.pow(tf.norm(w), 2.0)
  loss = tf.identity(avg_loss)
  loss += l2_regularizer_weight * weight_norm
  penalty_weight = (predefined_penalty_weight if step >= penalty_anneal_iters else 1.0)
  loss += penalty_weight * avg_penalty
  # rescale loss
  if penalty_weight > 1.0:
    loss /= penalty_weight
  return loss


 # ~~~~~ CLoVE ~~~~~ #

def laplacian_kernel(r1, r2, width=0.4):
  """Computes Laplacian kernel
  """
  return tf.math.exp(-tf.math.abs(tf.add(r1, -r2))/width)


def calibration_penalty(dist, y, kernel_width=0.001):
  """Computes the CLOvE penalty with Laplacian kernel
  """
  n = len(dist)
  y_pred = 1 - dist
  confidence = tf.math.maximum(y_pred, 1-y_pred)
  correctness = tf.cast(tf.math.abs(dist - tf.cast(y, tf.float32)) < 0.5, tf.float32)
  diffs = correctness - confidence

  # products of differences
  diffs_1 = tf.expand_dims(diffs, 0)
  diffs_2 = tf.expand_dims(diffs, 1)
  diffs_prod = tf.math.multiply(diffs_1, diffs_2)

  # kernel
  confidence_1 = tf.expand_dims(confidence, 0)
  confidence_2 = tf.expand_dims(confidence, 1)
  kernel = laplacian_kernel(confidence_1, confidence_2, kernel_width)
  
  return tf.math.sqrt(tf.reduce_mean(diffs_prod * kernel))


 # ~~~~~ VarAUC ~~~~~ #

def sigmoid(x, beta):
  """Steep sigmoid as a differential approximation of a step function
  """
  return 1/(1+tf.math.exp(-beta*x))


def soft_auc(dist, y, beta=10.0):
   """Differential approximation of AUC
   """
   dist, y = tf.reshape(dist, -1), tf.reshape(y, -1)
   n = len(dist)
   y_pred = 1 - dist

   pos = y_pred[y==1]
   neg = y_pred[y==0]
   d = tf.transpose([pos]) - neg

   sig = sigmoid(d, beta)
   return tf.reduce_sum(sig) / (len(neg) * len(pos))


# ~~~~~ Evaluation ~~~~~ #

def evaluate(g, z1, z2, y):
  """ AUC of a trained model
  """
  z1_hat = g(z1)
  z2_hat = g(z2)

  dists = cosine_distance(z1_hat, z2_hat)
  sims = 1 - dists

  fpr, tpr, thresholds = roc_curve(y, sims, pos_label=1)
  auc_val = auc(fpr, tpr)
  return auc_val


 # ~~~~~ Training ~~~~~ #

def training(optimizer, g, z_train, c_train, train_envs, val_z1, val_z2, val_y, test_z1, test_z2, test_y, n_pairs, pos_per_class, factor, n_sim_envs, penalty_type=None, l2_regularizer_weight=tf.constant(0.001)):
  """ Implementation of our algorithm with IRM, CLoVE, VarREX and VarAUC penalties. When no penalty is provided training is performed according to ERM.
  """  
  Nz, Nzs = 0, []
  losses = []
  val_aucs, test_aucs = [], []

  n_classes_in_env = len(train_envs[0])
  itrs = n_pairs // (n_sim_envs * n_classes_in_env * pos_per_class*2)

  for itr in range(itrs):

    sim_envs = [train_envs[i+itr*n_sim_envs] for i in range(n_sim_envs)]
    itr_penalties, itr_base_losses = [], []

    with tf.GradientTape() as tape:

        for e in sim_envs:
            env_bool = np.isin(c_train, e)
            env_x, env_y = z_train[env_bool], c_train[env_bool]
            env_z1, env_z2, env_y, env_Cs = make_pairs(env_x, env_y, pos_per_class)
            Nz += len(env_z1)

            # representations
            env_z1_hat = g(env_z1)
            env_z2_hat = g(env_z2)

            # env loss and penalty
            env_base_loss, dist = compute_loss(env_z1_hat, env_z2_hat, env_y)
            itr_base_losses.append(env_base_loss)

            if factor > 0:
                if penalty_type=='VarAUC':
                    env_penalty_val = soft_auc(dist, env_y)
                elif penalty_type=='CLoVE':
                    env_penalty_val = calibration_penalty(dist, env_y)
                elif penalty_type=='IRM':
                    env_penalty_val = IRM_penalty(env_z1_hat, env_z2_hat, env_y)
                elif penalty_type=='VarREx':
                    env_penalty_val = env_base_loss
            # ERM
            else: 
                env_penalty_val = 0.0
            itr_penalties.append(env_penalty_val)

        # loss
        avg_loss = tf.reduce_mean(itr_base_losses)
        if factor > 0:
            if penalty_type=='VarAUC':
                agg_penalty = tf.math.reduce_std(itr_penalties)
                loss = avg_loss + factor * agg_penalty
            elif penalty_type=='CLoVE':
                agg_penalty = tf.reduce_mean(itr_penalties)
                loss = avg_loss + factor * agg_penalty
            elif penalty_type=='IRM':
                agg_penalty = tf.reduce_mean(itr_penalties)
                loss = IRM_loss(g.trainable_weights, avg_loss, agg_penalty, itr, l2_regularizer_weight, factor)
            elif penalty_type=='VarREx':
                agg_penalty = tf.math.reduce_std(itr_penalties)
                loss = avg_loss + factor * agg_penalty
        # ERM
        else: 
            agg_penalty = 0.0
            loss = avg_loss

    grads = tape.gradient(loss, g.trainable_weights)
    optimizer.apply_gradients(zip(grads, g.trainable_weights))

    # save stats
    losses.append(loss)
    Nzs.append(Nz)
    auc_val = evaluate(g, val_z1, val_z2, val_y)
    val_aucs.append(auc_val)
    auc_test = evaluate(g, test_z1, test_z2, test_y)
    test_aucs.append(auc_test)

    if itr % 5 == 0:
      if factor > 0:
         print("{:.0%} completed, train loss {:.4f}, without penalty {:.4f}, penalty {:.4f}, validation auc {:.4f}, test auc {:.4f}".format(Nz/n_pairs, loss.numpy(), avg_loss.numpy(), agg_penalty.numpy(), auc_val, auc_test))
      else:
        print("{:.0%} completed, train loss {:.4f}, validation auc {:.4f}, test auc {:.4f}".format(Nz/n_pairs, loss.numpy(), auc_val, auc_test))

  tf.keras.backend.clear_session()
  return g, losses, Nzs, test_aucs, val_aucs
