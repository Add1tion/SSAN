"""
Ensemble prediction script for generating geochemical prospectivity maps.

This script loads trained TensorFlow 1.x checkpoint models from the K-fold
training process and applies them to every valid pixel location in the full
geochemical raster cube. For each pixel, a fixed-size spatial patch and its
center-pixel spectral vector are extracted and passed into each trained model.
The prediction probabilities from all available fold models are averaged to
produce an ensemble probability map for the positive class.

The script supports the same ablation switches used during training, including
causal feature ordering, geological constraint configuration, spectral attention,
spatial attention, and branch selection. These options must match the settings
used when the models were trained so that the correct checkpoint files can be
found and the TensorFlow graph can be rebuilt consistently.

Required input files:

* Geochemical `.tif` feature layers in `geochemical_data/`
* Trained model checkpoints in `D:\SSAN_Checkpoints`
* `causal_feature_order_custom.npy` if the model was trained with causal ordering

Example usage:

* Run ensemble prediction with default settings:
  `python predict_ensemble.py`

* Predict using a model trained without geological constraints:
  `python predict_ensemble.py --no_geo`

* Predict using a spatial-only model:
  `python predict_ensemble.py --branch spatial`

Main outputs:

* `predictions_<model_name>_ENSEMBLE.csv`, containing X/Y coordinates,
  positive-class probability, and predicted class for each valid pixel
* `area_fraction_plot_<model_name>_ENSEMBLE.png`, showing the relationship
  between probability threshold and cumulative predicted area, if georeferencing
  information and Matplotlib are available
  """

import tifffile
import os
import numpy as np
import tensorflow.compat.v1 as tf
import csv
import glob
from pathlib import Path
import argparse


try:
    import matplotlib.pyplot as plt

except ImportError:
    plt = None

tf.disable_v2_behavior()


def get_args():
    parser = argparse.ArgumentParser(description="Ensemble Prediction Script with Ablation Switches")
    parser.add_argument('--use_causal', action='store_true', default=True,
                        help='Model was trained with causal ordering.')
    parser.add_argument('--no_causal', action='store_false', dest='use_causal',
                        help='Model was not trained with causal ordering.')
    parser.add_argument('--use_geo', action='store_true', default=True,
                        help='Model was trained with geological constraints.')
    parser.add_argument('--no_geo', action='store_false', dest='use_geo',
                        help='Model was not trained with geological constraints.')
    parser.add_argument('--use_spectral_att', action='store_true', default=True,
                        help='Model was trained with spectral attention.')
    parser.add_argument('--no_spectral_att', action='store_false', dest='use_spectral_att',
                        help='Model was not trained with spectral attention.')
    parser.add_argument('--use_spatial_att', action='store_true', default=True,
                        help='Model was trained with spatial attention.')
    parser.add_argument('--no_spatial_att', action='store_false', dest='use_spatial_att',
                        help='Model was not trained with spatial attention.')
    parser.add_argument('--branch', type=str, default='full', choices=['full', 'spectral', 'spatial'],
                        help="The branch of the trained model.")
    args = parser.parse_args()
    return args



def load_and_prepare_data(args):
    TIF_DIR = Path('geochemical_data')

    try:
        tif_files = sorted(TIF_DIR.glob('*.tif'))
        if not tif_files:
            raise IndexError
        first_tif_path = tif_files[0]
    except IndexError:
        exit()

    with tifffile.TiffFile(first_tif_path) as tif:
        tags = tif.pages[0].tags
        scale = tags.get('ModelPixelScaleTag')
        tiepoint = tags.get('ModelTiepointTag')

        if scale and tiepoint:
            scale_val, tiepoint_val = scale.value, tiepoint.value
            pixel_w, pixel_h = scale_val[0], scale_val[1]
            x0, y0 = tiepoint_val[3], tiepoint_val[4]
            geo_info = (x0, y0, pixel_w, pixel_h)
        else:
            geo_info = None

    feature_cubes = [tifffile.imread(f) for f in tif_files]
    cube = np.stack(feature_cubes, axis=-1)

    if args.use_causal:
        original_feature_order = [f.stem for f in tif_files]
        causal_order_file = Path('causal_feature_order_custom.npy')

        if not causal_order_file.exists():
            exit()

        causal_feature_order = np.load(
            causal_order_file,
            allow_pickle=True
        ).tolist()

        original_index_map = {
            name.lower(): i
            for i, name in enumerate(original_feature_order)
        }

        try:
            reorder_indices = [
                original_index_map[name.lower()]
                for name in causal_feature_order
            ]
            cube = cube[:, :, reorder_indices]

        except KeyError:
            exit()

    return cube, geo_info

def build_tf_graph(window_size, num_components):

    OPTIMAL_NUM_HIDDEN = 128


    num_hidden, num_classes, ATTENTION_SIZE = OPTIMAL_NUM_HIDDEN, 2, 32
    timesteps, num_input = num_components, 1
    l2_regularizer = tf.keras.regularizers.l2(0.01)


    s1 = (window_size + 2 - 1) // 2
    s2 = ((s1 + 2 - 1) // 2)
    flattened_size = s2 * s2 * 64


    spe_weights = tf.Variable(tf.keras.initializers.he_normal()(shape=[num_hidden * 2, num_classes]),
                              name='spe_weights', regularizer=l2_regularizer)
    spe_biases = tf.Variable(tf.zeros_initializer()(shape=[num_classes]), name='spe_biases')
    spa_weights = {
        'wc1': tf.Variable(tf.keras.initializers.he_normal()(shape=[5, 5, num_components, 32]), name='wc1',
                           regularizer=l2_regularizer),
        'wc2': tf.Variable(tf.keras.initializers.he_normal()(shape=[5, 5, 32, 64]), name='wc2',
                           regularizer=l2_regularizer),
        'wd1': tf.Variable(tf.keras.initializers.he_normal()(shape=[flattened_size, 1024]), name='wd1',
                           regularizer=l2_regularizer),
        'out': tf.Variable(tf.keras.initializers.he_normal()(shape=[1024, num_classes]), name='spa_out',
                           regularizer=l2_regularizer)
    }
    spa_biases = {
        'bc1': tf.Variable(tf.zeros_initializer()(shape=[32]), name='bc1'),
        'bc2': tf.Variable(tf.zeros_initializer()(shape=[64]), name='bc2'),
        'bd1': tf.Variable(tf.zeros_initializer()(shape=[1024]), name='bd1'),
        'out': tf.Variable(tf.zeros_initializer()(shape=[num_classes]), name='spa_bd_out')
    }
    m_weights = {
        'wf1': tf.Variable(tf.keras.initializers.he_normal()(shape=[num_classes, 256]), name='wf1',
                           regularizer=l2_regularizer),
        'wf1_spa': tf.Variable(tf.keras.initializers.he_normal()(shape=[num_classes, 256]), name='wf1_spa',
                               regularizer=l2_regularizer),
        'wf2': tf.Variable(tf.keras.initializers.he_normal()(shape=[512, 1024]), name='wf2',
                           regularizer=l2_regularizer),
        'mout': tf.Variable(tf.keras.initializers.he_normal()(shape=[1024, num_classes]), name='mout',
                            regularizer=l2_regularizer)
    }
    m_biases = {
        'bf1': tf.Variable(tf.zeros_initializer()(shape=[256]), name='bf1'),
        'bf1_spa': tf.Variable(tf.zeros_initializer()(shape=[256]), name='bf1_spa'),
        'bf2': tf.Variable(tf.zeros_initializer()(shape=[1024]), name='bf2'),
        'mout': tf.Variable(tf.zeros_initializer()(shape=[num_classes]), name='bf_out')
    }


    spe_X = tf.placeholder(tf.float32, [None, timesteps, num_input], name='spe_X')
    spa_X = tf.placeholder(tf.float32, [None, window_size, window_size, num_components], name='spa_X')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(tf.bool, name='is_training')

    def SpectralAttention(inputs, attention_size, regularizer):
        if isinstance(inputs, tuple): inputs = tf.concat(inputs, 2)
        hidden_size = inputs.shape[2].value
        w_omega = tf.get_variable('w_omega', [hidden_size, attention_size],
                                  initializer=tf.random_normal_initializer(stddev=0.1), regularizer=regularizer)
        b_omega = tf.get_variable('b_omega', [attention_size], initializer=tf.random_normal_initializer(stddev=0.1),
                                  regularizer=regularizer)
        u_omega = tf.get_variable('u_omega', [attention_size], initializer=tf.random_normal_initializer(stddev=0.1),
                                  regularizer=regularizer)
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1)
        alphas = tf.nn.softmax(vu, name="attention_weights")
        return tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    def ARNN(x, use_attention=True):
        gru_fw = tf.nn.rnn_cell.GRUCell(num_hidden)
        gru_bw = tf.nn.rnn_cell.GRUCell(num_hidden)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(gru_fw, gru_bw, x, dtype=tf.float32)
        if use_attention:
            with tf.variable_scope("SpectralAttention"):
                a_out = SpectralAttention(outputs, ATTENTION_SIZE, regularizer=l2_regularizer)
        else:
            a_out = tf.concat(final_states, axis=-1)
        return tf.nn.xw_plus_b(a_out, spe_weights, spe_biases)

    def SpatialAttention(feature_map, regularizer, scope="SpatialAttention"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            _, H, W, C = feature_map.get_shape().as_list()
            w_s = tf.get_variable("w_s", [C, 1], initializer=tf.truncated_normal_initializer(stddev=0.0001),
                                  regularizer=regularizer)
            b_s = tf.get_variable("b_s", [1], initializer=tf.zeros_initializer(), regularizer=regularizer)
            flat = tf.reshape(feature_map, [-1, C])
            sa_logits = tf.matmul(flat, w_s) + b_s
            sa_logits_reshaped = tf.reshape(sa_logits, [-1, H * W])
            sa = tf.nn.sigmoid(sa_logits_reshaped, name="attention_weights")
            att = tf.reshape(tf.concat([sa] * C, axis=1), [-1, H, W, C])
            return att * feature_map

    def ACNN(x, use_attention=True):
        x = tf.reshape(x, [-1, window_size, window_size, num_components])
        x_input_for_conv = SpatialAttention(x, regularizer=l2_regularizer,
                                            scope="SpatialInputAttention") if use_attention else x
        conv1 = tf.nn.relu(tf.nn.conv2d(x_input_for_conv, spa_weights['wc1'], [1, 1, 1, 1], 'SAME') + spa_biases['bc1'])
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, spa_weights['wc2'], [1, 1, 1, 1], 'SAME') + spa_biases['bc2'])
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        flat = tf.layers.flatten(pool2)
        fc1 = tf.nn.relu(tf.matmul(flat, spa_weights['wd1']) + spa_biases['bd1'])
        fc1 = tf.nn.dropout(fc1, keep_prob)
        return tf.matmul(fc1, spa_weights['out']) + spa_biases['out']

    def SSAN_1_model(spe_x, spa_x, branch_to_use='full', use_spectral_att=True, use_spatial_att=True):
        with tf.variable_scope("ACNN"):
            spa_logit = ACNN(spa_x, use_attention=use_spatial_att)
        spe_logit = ARNN(spe_x, use_attention=use_spectral_att)
        if branch_to_use == 'spectral': return spe_logit
        if branch_to_use == 'spatial': return spa_logit
        spe_logit_bn = tf.layers.batch_normalization(spe_logit, training=is_training)
        spa_logit_bn = tf.layers.batch_normalization(spa_logit, training=is_training)
        spe_fc = tf.nn.dropout(tf.nn.relu(tf.matmul(spe_logit_bn, m_weights['wf1']) + m_biases['bf1']), keep_prob)
        spa_fc = tf.nn.dropout(tf.nn.relu(tf.matmul(spa_logit_bn, m_weights['wf1_spa']) + m_biases['bf1_spa']),
                               keep_prob)
        merge = tf.concat([spe_fc, spa_fc], axis=1)
        m2 = tf.nn.relu(tf.matmul(merge, m_weights['wf2']) + m_biases['bf2'])
        return tf.matmul(m2, m_weights['mout']) + m_biases['mout']


    logits = SSAN_1_model(spe_X, spa_X, args.branch, args.use_spectral_att, args.use_spatial_att)
    prediction_tensor = tf.nn.softmax(logits, name="Softmax")

    return prediction_tensor


def run_ensemble_prediction(args, cube, geo_info, prediction_tensor):
    H, W, num_components = cube.shape
    window_size = 9
    half = window_size // 2


    save_dir = r'D:\SSAN_Checkpoints'
    base_model_name = (f"ssan_branch-{args.branch}_causal-{args.use_causal}_geo-{args.use_geo}"
                       f"_specATT-{args.use_spectral_att}_spatATT-{args.use_spatial_att}")


    model_search_pattern = os.path.join(save_dir, f"{base_model_name}_best_fold-*.ckpt.meta")
    meta_files = glob.glob(model_search_pattern)

    if not meta_files:
        exit()

    model_paths = [path.replace('.meta', '') for path in meta_files]

    all_prob_maps = []

    for i, model_path in enumerate(model_paths):

        tf.reset_default_graph()

        prediction_tensor = build_tf_graph(window_size, num_components)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            graph = tf.get_default_graph()
            spe_X = graph.get_tensor_by_name('spe_X:0')
            spa_X = graph.get_tensor_by_name('spa_X:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            is_training = graph.get_tensor_by_name('is_training:0')

            current_prob_map = np.zeros((H, W), dtype=np.float32)
            total_pixels = (H - 2 * half) * (W - 2 * half)
            processed_pixels = 0

            for r in range(half, H - half):
                for c in range(half, W - half):
                    patch = cube[r - half:r + half + 1, c - half:c + half + 1, :]
                    center_spec = patch[half, half, :].reshape(num_components, 1)

                    spa_batch = np.expand_dims(patch, axis=0)
                    spe_batch = np.expand_dims(center_spec, axis=0)

                    feed_dict = {
                        spe_X: spe_batch,
                        spa_X: spa_batch,
                        keep_prob: 1.0,
                        is_training: False
                    }

                    probs = sess.run(prediction_tensor, feed_dict=feed_dict)
                    current_prob_map[r, c] = probs[0, 1]

                processed_pixels += (W - 2 * half)
                if (r - half + 1) % 50 == 0:
                    print(
                        f"    ... {i + 1}  {processed_pixels}/{total_pixels}  ({processed_pixels / total_pixels:.1%})")

            all_prob_maps.append(current_prob_map)

    ensembled_prob_map = np.mean(all_prob_maps, axis=0)

    return ensembled_prob_map, base_model_name


def save_results(ensembled_prob_map, geo_info, base_model_name):

    H, W = ensembled_prob_map.shape
    window_size = 9
    half = window_size // 2

    csv_rows = []
    for r in range(half, H - half):
        for c in range(half, W - half):
            prob_1 = ensembled_prob_map[r, c]
            pred_cls = 1 if prob_1 >= 0.4 else 0

            if geo_info:
                x0, y0, pixel_w, pixel_h = geo_info
                X_coord = x0 + (c + 0.5) * pixel_w
                Y_coord = y0 - (r + 0.5) * pixel_h
            else:
                X_coord, Y_coord = c, r

            csv_rows.append([X_coord, Y_coord, float(prob_1), int(pred_cls)])

    output_csv_path = f'predictions_{base_model_name}_ENSEMBLE.csv'
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'prob_1', 'pred_class'])
        writer.writerows(csv_rows)

    # --- 绘图 ---
    if plt is None:
        return

    if not geo_info:
        return

    probs_class_1 = np.array([row[2] for row in csv_rows])
    _, _, pixel_w, pixel_h = geo_info
    pixel_area = abs(pixel_w * pixel_h)
    total_area = len(csv_rows) * pixel_area

    if total_area == 0:
        return

    thresholds = np.linspace(1, 0, 101)
    area_fractions = [(np.sum(probs_class_1 >= t) * pixel_area) / total_area for t in thresholds]

    plt.figure(figsize=(10, 8))
    plt.plot(area_fractions, thresholds, color='blue', linewidth=2)
    plt.title('Threshold vs. Cumulative Area Fraction (Ensemble Prediction)', fontsize=16)
    plt.xlabel('Cumulative Area Fraction', fontsize=12)
    plt.ylabel('Probability Threshold', fontsize=12)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().invert_yaxis()

    output_curve_path = f'area_fraction_plot_{base_model_name}_ENSEMBLE.png'
    plt.savefig(output_curve_path, dpi=300)


if __name__ == "__main__":
    args = get_args()
    cube, geo_info = load_and_prepare_data(args)


    window_size = 9
    _, _, num_components = cube.shape
    prediction_tensor_def = build_tf_graph(window_size, num_components)

    ensembled_map, model_name_base = run_ensemble_prediction(args, cube, geo_info, prediction_tensor_def)
    save_results(ensembled_map, geo_info, model_name_base)






