import numpy as np
import vis_util
import transform_util as tr_util

def get_view_norm_skeleton_data(pred_15_fit):
    global_angles = []
    pred_15_view_norm = []
    pred_15_local = []

    for skeleton in pred_15_fit:
        alpha, beta, gamma, view_norm_skeleton = tr_util.get_euler_angles_and_transform(skeleton)
        local_limbs = tr_util.prior_global2local(view_norm_skeleton)

        global_angles.append([alpha, beta, gamma])
        pred_15_view_norm.append(view_norm_skeleton)
        pred_15_local.append(local_limbs)

    return np.array(global_angles), np.array(pred_15_view_norm), np.array(pred_15_local)

def convert_x19_batch_to_x15_batch(x19_batch):
    # Shape: [B, T, 19, 3]
    x15_batch = np.zeros((x19_batch.shape[0], x19_batch.shape[1], 15, 3))
    for i in range(len(x19_batch)):
        x15_batch[i] = convert_x19_seq_to_x15_seq(x19_batch[i])
    return x15_batch

def convert_x19_seq_to_x15_seq(x_19_seq):
    # Input Shape: [T, 19, 3]
    # Output Shape: [T, 15, 3]
    x_18_batch = np.zeros((x_19_seq.shape[0], 18, 3))
    x_18_batch[:, :15] = x_19_seq[:, :15]
    x_18_batch[:, 15] = np.arctan2(x_19_seq[:, 18], x_19_seq[:, 15])
    x_15_batch = get_global_skeleton_data(x_18_batch)
    return x_15_batch

def get_global_skeleton_data(pred_18_local):
    pred_15_local = pred_18_local[:, :15]
    global_angles = pred_18_local[:, 15]

    global_skeletons = []
    for i in range(len(pred_18_local)):
        view_norm_skeleton = tr_util.prior_local2global(pred_15_local[i])
        alpha, beta, gamma = global_angles[i]
        global_skeleton = tr_util.get_global_joints(view_norm_skeleton, alpha, beta, gamma)
        global_skeletons.append(global_skeleton)
    return np.array(global_skeletons)

def get_skeleton_images(x15_seq, title_prefix=''):
    return [vis_util.plot_skeleton(x15_seq[i], title=(title_prefix + ' Frame %4d' % i)) for i in range(len(x15_seq))]

