import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import shutil
import math
import scipy.io

# Setup GPU for training
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CPU:-1; GPU0: 1; GPU1: 0;
np.random.seed(1111)
tf.set_random_seed(1111)

class DeepElasticWave:
    # Initialize the class
    def __init__(self, Collo, SRC, IC, FIXED, DIST, uv_layers, dist_layers, part_layers, lb, ub, uvDir='', partDir='', distDir=''):

        # Count for callback function
        self.count = 0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.E = 2.5
        self.mu = 0.25
        self.rho = 1.0

        # P wave velocity: sqrt((lam+2nu)/rho)=1.732

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        self.t_c = Collo[:, 2:3]

        # Source wave
        self.x_SRC = SRC[:, 0:1]
        self.y_SRC = SRC[:, 1:2]
        self.t_SRC = SRC[:, 2:3]
        self.u_SRC = SRC[:, 3:4]
        self.v_SRC = SRC[:, 4:5]

        # Initial condition point, t=0
        self.x_IC = IC[:, 0:1]
        self.y_IC = IC[:, 1:2]
        self.t_IC = IC[:, 2:3]

        self.x_FIX = FIXED[:, 0:1]
        self.y_FIX = FIXED[:, 1:2]
        self.t_FIX = FIXED[:, 2:3]

        self.x_dist = DIST[:, 0:1]
        self.y_dist = DIST[:, 1:2]
        self.t_dist = DIST[:, 2:3]
        self.u_dist = DIST[:, 3:4]
        self.v_dist = DIST[:, 4:5]
        self.s11_dist = DIST[:, 5:6]
        self.s22_dist = DIST[:, 6:7]
        self.s12_dist = DIST[:, 7:8]

        # Define layers
        self.uv_layers = uv_layers
        self.dist_layers = dist_layers
        self.part_layers = part_layers

        # Initialize NNs
        if uvDir is not '':
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)
        else:
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)

        if distDir is not '':
            print("Loading dist NN ...")
            self.dist_weights, self.dist_biases = self.load_NN(distDir, self.dist_layers)
        else:
            self.dist_weights, self.dist_biases = self.initialize_NN(self.dist_layers)

        if partDir is not '':
            print("Loading part NN ...")
            self.part_weights, self.part_biases = self.load_NN(partDir, self.part_layers)
        else:
            self.part_weights, self.part_biases = self.initialize_NN(self.part_layers)


        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x_c.shape[1]])    # Point for postprocessing
        self.y_tf = tf.placeholder(tf.float64, shape=[None, self.y_c.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_c.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float64, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float64, shape=[None, self.y_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float64, shape=[None, self.t_c.shape[1]])

        self.x_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.x_SRC.shape[1]])
        self.y_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.y_SRC.shape[1]])
        self.t_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.t_SRC.shape[1]])
        self.u_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.u_SRC.shape[1]])
        self.v_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.v_SRC.shape[1]])

        self.x_IC_tf = tf.placeholder(tf.float64, shape=[None, self.x_IC.shape[1]])
        self.y_IC_tf = tf.placeholder(tf.float64, shape=[None, self.y_IC.shape[1]])
        self.t_IC_tf = tf.placeholder(tf.float64, shape=[None, self.t_IC.shape[1]])

        self.x_FIX_tf = tf.placeholder(tf.float64, shape=[None, self.x_FIX.shape[1]])
        self.y_FIX_tf = tf.placeholder(tf.float64, shape=[None, self.y_FIX.shape[1]])
        self.t_FIX_tf = tf.placeholder(tf.float64, shape=[None, self.t_FIX.shape[1]])

        self.x_dist_tf = tf.placeholder(tf.float64, shape=[None, self.x_dist.shape[1]])
        self.y_dist_tf = tf.placeholder(tf.float64, shape=[None, self.y_dist.shape[1]])
        self.t_dist_tf = tf.placeholder(tf.float64, shape=[None, self.t_dist.shape[1]])
        self.u_dist_tf = tf.placeholder(tf.float64, shape=[None, self.u_dist.shape[1]])
        self.v_dist_tf = tf.placeholder(tf.float64, shape=[None, self.v_dist.shape[1]])
        self.s11_dist_tf = tf.placeholder(tf.float64, shape=[None, self.s11_dist.shape[1]])
        self.s22_dist_tf = tf.placeholder(tf.float64, shape=[None, self.s22_dist.shape[1]])
        self.s12_dist_tf = tf.placeholder(tf.float64, shape=[None, self.s12_dist.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.ut_pred, self.vt_pred, self.s11_pred, self.s22_pred, self.s12_pred = self.net_uv(self.x_tf, self.y_tf, self.t_tf)
        self.e11_pred, self.e22_pred, self.e12_pred = self.net_e(self.x_tf, self.y_tf, self.t_tf)

        self.u_IC_pred, self.v_IC_pred, self.ut_IC_pred, self.vt_IC_pred, _, _, _ = self.net_uv(self.x_IC_tf, self.y_IC_tf, self.t_IC_tf)
        self.u_FIX_pred, self.v_FIX_pred, _, _, _, _, _ = self.net_uv(self.x_FIX_tf, self.y_FIX_tf, self.t_FIX_tf)
        # self.ut_IC_pred, self.vt_IC_pred = self.net_vel(self.x_IC_tf, self.y_IC_tf, self.t_IC_tf)
        self.u_SRC_pred, self.v_SRC_pred, _, _, _, _, _ = self.net_uv(self.x_SRC_tf, self.y_SRC_tf, self.t_SRC_tf)

        self.f_pred_u, self.f_pred_v, self.f_pred_ut, self.f_pred_vt, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12 = self.net_f_sig(self.x_c_tf, self.y_c_tf, self.t_c_tf)

        # Not included in training, just for evaluation
        self.loss_SRC = tf.reduce_mean(tf.square(self.u_SRC_pred-self.u_SRC_tf)) \
                         + tf.reduce_mean(tf.square(self.v_SRC_pred-self.v_SRC_tf))
        self.loss_IC = tf.reduce_mean(tf.square(self.u_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.v_IC_pred))\
                         + tf.reduce_mean(tf.square(self.ut_IC_pred))\
                         + tf.reduce_mean(tf.square(self.vt_IC_pred))
        self.loss_FIX = tf.reduce_mean(tf.square(self.u_FIX_pred)) \
                         + tf.reduce_mean(tf.square(self.v_FIX_pred))

        self.loss_f_uv = tf.reduce_mean(tf.square(self.f_pred_u)) \
                         + tf.reduce_mean(tf.square(self.f_pred_v))\
                         + tf.reduce_mean(tf.square(self.f_pred_ut))\
                         + tf.reduce_mean(tf.square(self.f_pred_vt))
        self.loss_f_s = tf.reduce_mean(tf.square(self.f_pred_s11)) \
                        + tf.reduce_mean(tf.square(self.f_pred_s22)) \
                        + tf.reduce_mean(tf.square(self.f_pred_s12))

        self.loss = 5*self.loss_f_uv + 5*self.loss_f_s + self.loss_SRC + self.loss_IC + self.loss_FIX

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    def save_NN(self, fileDir, TYPE=''):
        if TYPE=='UV':
            uv_weights = self.sess.run(self.uv_weights)
            uv_biases = self.sess.run(self.uv_biases)
        elif TYPE=='DIST':
            uv_weights = self.sess.run(self.dist_weights)
            uv_biases = self.sess.run(self.dist_biases)
        elif TYPE=='PART':
            uv_weights = self.sess.run(self.part_weights)
            uv_biases = self.sess.run(self.part_biases)
        else:
            pass
        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save "+ TYPE+ " NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            # print(len(uv_weights))
            # print(np.shape(uv_weights))
            # print(num_layers)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float64)
                b = tf.Variable(uv_biases[num], dtype=tf.float64)
                weights.append(W)
                biases.append(b)
                print("Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_dist(self, x, y, t):
        dist = self.neural_net(tf.concat([x, y, t], 1), self.dist_weights, self.dist_biases)
        D_u = dist[:,0:1]
        D_v = dist[:, 1:2]
        D_s11 = dist[:, 2:3]
        D_s22 = dist[:, 3:4]
        D_s12 = dist[:, 4:5]
        return D_u, D_v, D_s11, D_s22, D_s12

    def net_dist_dt(self, x, y, t):
        # Time derivative of distance function
        dist = self.neural_net(tf.concat([x, y, t], 1), self.dist_weights, self.dist_biases)
        D_u = dist[:, 0:1]
        D_v = dist[:, 1:2]
        D_s11 = dist[:, 2:3]
        D_s22 = dist[:, 3:4]
        D_s12 = dist[:, 4:5]

        dt_D_u = tf.gradients(D_u, t)[0]
        dt_D_v = tf.gradients(D_v, t)[0]
        dt_D_s11 = tf.gradients(D_s11, t)[0]
        dt_D_s22 = tf.gradients(D_s22, t)[0]
        dt_D_s12 = tf.gradients(D_s12, t)[0]

        return dt_D_u, dt_D_v, dt_D_s11, dt_D_s22, dt_D_s12

    def net_part(self, x, y, t):
        P = self.neural_net(tf.concat([x, y, t], 1), self.part_weights, self.part_biases)
        P_u = P[:,0:1]
        P_v = P[:, 1:2]
        P_s11 = P[:, 2:3]
        P_s22 = P[:, 3:4]
        P_s12 = P[:, 4:5]
        dt_P_u = tf.gradients(P_u, t)[0]
        dt_P_v = tf.gradients(P_v, t)[0]
        return P_u, P_v, P_s11, P_s22, P_s12, dt_P_u, dt_P_v

    def net_uv(self, x, y, t):
        # This NN return sigma_phi
        uv_sig = self.neural_net(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)

        u = uv_sig[:, 0:1]
        v = uv_sig[:, 1:2]
        ut = uv_sig[:, 2:3]
        vt = uv_sig[:, 3:4]
        s11 = uv_sig[:, 4:5]
        s22 = uv_sig[:, 5:6]
        s12 = uv_sig[:, 6:7]

        return u, v, ut, vt, s11, s22, s12

    def net_e(self, x, y, t):
        u, v, _, _, _, _, _ = self.net_uv(x, y, t)
        # Strains
        e11 = tf.gradients(u, x)[0]
        e22 = tf.gradients(v, y)[0]
        e12 = (tf.gradients(u, y)[0] + tf.gradients(v, x)[0])
        return e11, e22, e12

    def net_f_sig(self, x, y, t):

        E = self.E
        mu = self.mu
        rho = self.rho

        u, v, ut, vt, s11, s22, s12 = self.net_uv(x, y, t)

        # Strains
        e11, e22, e12 = self.net_e(x, y, t)

        # Plane stress problem
        # sp11 = E / (1 - mu * mu) * e11 + E * mu / (1 - mu * mu) * e22
        # sp22 = E * mu / (1 - mu * mu) * e11 + E / (1 - mu * mu) * e22
        # sp12 = E / (2 * (1 + mu)) * e12

        # Plane strain problem
        coef = E/((1+mu)*(1-2*mu))
        sp11 = coef * (1-mu) * e11 + coef * mu * e22
        sp22 = coef * mu * e11 + coef * (1-mu) * e22
        sp12 = E / (2 * (1 + mu)) * e12

        # Cauchy stress
        f_s11 = s11 - sp11
        f_s12 = s12 - sp12
        f_s22 = s22 - sp22

        f_ut = tf.gradients(u, t)[0]-ut
        f_vt = tf.gradients(v, t)[0]-vt

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        # u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(ut, t)[0]

        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]
        # v_t = tf.gradients(v, t)[0]
        v_tt = tf.gradients(vt, t)[0]

        # f_u:=Sxx_x+Sxy_y-rho*u_tt
        f_u = s11_1 + s12_2 - rho*u_tt
        f_v = s22_2 + s12_1 - rho*v_tt

        return f_u, f_v, f_ut, f_vt, f_s11, f_s22, f_s12

    def net_surf_var(self, x, y, t, nx, ny):
        # In our case, the nx, ny for one edge is same
        # Return surface traction tx, ty

        u, v, _, _, s11, s22, s12 = self.net_uv(x, y, t)

        tx = tf.multiply(s11, nx) + tf.multiply(s12, ny)
        ty = tf.multiply(s12, nx) + tf.multiply(s22, ny)

        return tx, ty

    def callback(self, loss):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def callback_dist(self, loss_dist):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss_dist))

    def callback_part(self, loss_part):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss_part))

    def train(self, iter, learning_rate, batch_num):

        loss_f_uv = []
        loss_f_s = []
        loss = []

        # The collocation point is splited into partitions of batch_num，1 epoch for training
        for i in range(batch_num):
            col_num = self.x_c.shape[0]
            idx_start = int(i * col_num / batch_num)
            idx_end = int((i + 1) * col_num / batch_num)

            tf_dict = {self.x_c_tf: self.x_c[idx_start:idx_end,:], self.y_c_tf: self.y_c[idx_start:idx_end,:], self.t_c_tf: self.t_c[idx_start:idx_end,:],
                       self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                       self.x_SRC_tf: self.x_SRC, self.y_SRC_tf: self.y_SRC, self.t_SRC_tf: self.t_SRC, self.u_SRC_tf: self.u_SRC, self.v_SRC_tf: self.v_SRC,
                       self.x_FIX_tf: self.x_FIX, self.y_FIX_tf: self.y_FIX, self.t_FIX_tf: self.t_FIX,
                       self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist, self.u_dist_tf: self.u_dist,
                       self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist, self.s12_dist_tf: self.s12_dist,
                       self.learning_rate: learning_rate}

            for it in range(iter):

                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e' %
                          (it, loss_value))

                loss_f_uv.append(self.sess.run(self.loss_f_uv, tf_dict))
                loss_f_s.append(self.sess.run(self.loss_f_s, tf_dict))

                loss.append(self.sess.run(self.loss, tf_dict))

        return loss_f_uv, loss_f_s, loss

    def train_bfgs(self, batch_num):
        # The collocation point is splited into partitions of batch_num
        for i in range(batch_num):
            col_num = self.x_c.shape[0]
            idx_start = int(i*col_num/batch_num)
            idx_end = int((i+1)*col_num/batch_num)
            tf_dict = {self.x_c_tf: self.x_c[idx_start:idx_end,:], self.y_c_tf: self.y_c[idx_start:idx_end,:], self.t_c_tf: self.t_c[idx_start:idx_end,:],
                       self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                       self.x_SRC_tf: self.x_SRC, self.y_SRC_tf: self.y_SRC, self.t_SRC_tf: self.t_SRC, self.u_SRC_tf: self.u_SRC, self.v_SRC_tf: self.v_SRC,
                       self.x_FIX_tf: self.x_FIX, self.y_FIX_tf: self.y_FIX, self.t_FIX_tf: self.t_FIX,
                       self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist, self.u_dist_tf: self.u_dist,
                       self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist, self.s12_dist_tf: self.s12_dist}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss],
                                    loss_callback=self.callback)


    def predict(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s11_star = self.sess.run(self.s11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s22_star = self.sess.run(self.s22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s12_star = self.sess.run(self.s12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e11_star = self.sess.run(self.e11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e22_star = self.sess.run(self.e22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e12_star = self.sess.run(self.e12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})

        return u_star, v_star, s11_star, s22_star, s12_star, e11_star, e22_star, e12_star

    def probe(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s11_star = self.sess.run(self.s11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s22_star = self.sess.run(self.s22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s12_star = self.sess.run(self.s12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e11_star = self.sess.run(self.e11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e22_star = self.sess.run(self.e22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e12_star = self.sess.run(self.e12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})

        return u_star, v_star, s11_star, s22_star, s12_star, e11_star, e22_star, e12_star

    def getloss(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                       self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                       self.x_SRC_tf: self.x_SRC, self.y_SRC_tf: self.y_SRC, self.t_SRC_tf: self.t_SRC, self.u_SRC_tf: self.u_SRC, self.v_SRC_tf: self.v_SRC,
                       self.x_FIX_tf: self.x_FIX, self.y_FIX_tf: self.y_FIX, self.t_FIX_tf: self.t_FIX,
                       self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist, self.u_dist_tf: self.u_dist,
                       self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist, self.s12_dist_tf: self.s12_dist}

        loss_SRC = self.sess.run(self.loss_SRC, tf_dict)
        loss_FIX = self.sess.run(self.loss_FIX, tf_dict)
        loss_IC = self.sess.run(self.loss_IC, tf_dict)
        loss_f_uv = self.sess.run(self.loss_f_uv, tf_dict)
        loss_f_s = self.sess.run(self.loss_f_s, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)

        print('loss_f_uv', loss_f_uv)
        print('loss_f_s', loss_f_s)
        print('loss_SRC', loss_SRC)
        print('loss_IC', loss_IC)
        print('loss_FIX', loss_FIX)
        print('loss', loss)


def GenDistPt(xmin, xmax, ymin, ymax, tmin, tmax, xc, yc, r, num_surf_pt, num, num_t):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)

    # Delete point in hole
    dst = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
    x = x[dst >= r]
    y = y[dst >= r]
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]

    # Add point on hole surface
    theta = np.linspace(0.0, np.pi*2, num_surf_pt)
    x_surf = np.multiply(r, np.cos(theta)) + xc
    y_surf = np.multiply(r, np.sin(theta)) + yc
    x_surf = x_surf.flatten()[:, None]
    y_surf = y_surf.flatten()[:, None]

    # Concatenate
    x = np.concatenate((x, x_surf), 0)
    y = np.concatenate((y, y_surf), 0)

    t = np.linspace(tmin, tmax, num=num_t)
    xxx, ttt = np.meshgrid(x, t)
    yyy, _ = np.meshgrid(y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt

def GenDist(XYT_dist):
    dist_u = np.zeros_like(XYT_dist[:,0:1])
    dist_v = np.zeros_like(XYT_dist[:,0:1])
    dist_s11 = np.zeros_like(XYT_dist[:,0:1])
    dist_s22 = np.zeros_like(XYT_dist[:,0:1])
    dist_s12 = np.zeros_like(XYT_dist[:,0:1])
    for i in range(len(XYT_dist)):
        dist_u[i, 0] = min(XYT_dist[i][2], ((XYT_dist[i][0])**2+(XYT_dist[i][1])**2)**0.5-2.0,
                           15-XYT_dist[i][0], XYT_dist[i][0]+15, 15-XYT_dist[i][1], XYT_dist[i][1]+15)/10.0
        dist_v[i, 0] = min(XYT_dist[i][2], ((XYT_dist[i][0])**2+(XYT_dist[i][1])**2)**0.5-2.0,
                           15-XYT_dist[i][0], XYT_dist[i][0]+15, 15-XYT_dist[i][1], XYT_dist[i][1]+15)/10.0
        dist_s11[i, 0] = 1.0  # unused
        dist_s22[i, 0] = 1.0
        dist_s12[i, 0] = 1.0

    DIST = np.concatenate((XYT_dist, dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return DIST

def CartGrid(xmin, xmax, ymin, ymax, tmin, tmax, num, num_t):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    xx, yy = np.meshgrid(x, y)
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, yyy, ttt = np.meshgrid(x, y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt

def preprocess(dir):
    # dir: directory of training data
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    U = data['u']
    V = data['v']
    Amp = data['amp']
    S11 = data['s11']
    S22 = data['s22']
    S12 = data['s12']
    Mis = data['Mises']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    u_star = U.flatten()[:, None]
    v_star = V.flatten()[:, None]
    a_star = Amp.flatten()[:, None]
    s11_star = S11.flatten()[:, None]
    s22_star = S22.flatten()[:, None]
    s12_star = S12.flatten()[:, None]
    mis_star = Mis.flatten()[:, None]

    return x_star, y_star, u_star, v_star, a_star, s11_star, s22_star, s12_star, mis_star


import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('seismic')
new_cmap = truncate_colormap(cmap, 0.5, 1.0)

def postProcess(xmin, xmax, ymin, ymax, field=[], s=12, num=0, dpi=100):
    ''' num: Number of time step
    '''

    [x_star, y_star, u_star, v_star, a_star, s11_star, s22_star, s12_star, _] = preprocess(
        '../FEM_result/30x30_gauss_fine/ProbeData-' + str(num) + '.mat')
    [x_pred, y_pred, t_pred, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.1)

    cf = ax[0, 0].scatter(x_pred, y_pred, c=u_pred, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s, vmin=-0.5, vmax=0.5)
    ax[0, 0].axis('square')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'$u$-PINN', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 0], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 0].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s, vmin=-0.5, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin+15, xmax+15])
    ax[1, 0].set_ylim([ymin+15, ymax+15])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_title(r'$u$-FEM', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 0], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[0, 1].scatter(x_pred, y_pred, c=v_pred, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s, vmin=-0.5, vmax=0.5)
    ax[0, 1].axis('square')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$v$-PINN', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 1], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s, vmin=-0.5, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin+15, xmax+15])
    ax[1, 1].set_ylim([ymin+15, ymax+15])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 1].set_title(r'$v$-FEM', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 1], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[0, 2].scatter(x_pred, y_pred, c=amp_pred, alpha=0.9, edgecolors='none', cmap=new_cmap, marker='o', s=s, vmin=0, vmax=0.5)
    ax[0, 2].axis('square')
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 2].set_title('Mag.-PINN', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 2], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 2].scatter(x_star, y_star, c=a_star, alpha=0.9, edgecolors='none', cmap=new_cmap, marker='o', s=s, vmin=0, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_xlim([xmin+15, xmax+15])
    ax[1, 2].set_ylim([ymin+15, ymax+15])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 2].set_title('Mag.-FEM', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 2], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    # plt.draw()
    plt.savefig('./output/uv_comparison_'+str(num).zfill(3)+'.png',dpi=dpi)
    plt.close('all')

    # Plot predicted stress
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.1)

    cf = ax[0, 0].scatter(x_pred, y_pred, c=s11_pred, alpha=0.9, edgecolors='none', marker='o', cmap='seismic', s=s, vmin=-1.5, vmax=1.5)
    ax[0, 0].axis('square')
    # for key, spine in ax[0, 0].spines.items():
    #     if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    #         spine.set_visible(False)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'$\sigma_{11}$-PINN', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 0], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 0].scatter(x_star, y_star, c=s11_star, alpha=0.9, edgecolors='none', marker='s', cmap='seismic', s=s, vmin=-1.5, vmax=1.5)
    ax[1, 0].axis('square')
    # for key, spine in ax[1, 0].spines.items():
    #     if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    #         spine.set_visible(False)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin+15, xmax+15])
    ax[1, 0].set_ylim([ymin+15, ymax+15])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_title(r'$\sigma_{11}$-FEM', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 0], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[0, 1].scatter(x_pred, y_pred, c=s22_pred, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s, vmin=-1.5, vmax=1.5)
    ax[0, 1].axis('square')
    # for key, spine in ax[0, 1].spines.items():
    #     if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    #         spine.set_visible(False)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title(r'$\sigma_{22}$-PINN', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 1], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 1].scatter(x_star, y_star, c=s22_star, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s, vmin=-1.5, vmax=1.5)
    ax[1, 1].axis('square')
    # for key, spine in ax[1, 1].spines.items():
    #     if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    #         spine.set_visible(False)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin+15, xmax+15])
    ax[1, 1].set_ylim([xmin+15, xmax+15])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 1].set_title(r'$\sigma_{22}$-FEM', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 1], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[0, 2].scatter(x_pred, y_pred, c=s12_pred, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s, vmin=-1.5, vmax=1.5)
    ax[0, 2].axis('square')
    # for key, spine in ax[0, 2].spines.items():
    #     if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    #         spine.set_visible(False)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 2].set_title(r'$\sigma_{12}$-PINN', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 2], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 2].scatter(x_star, y_star, c=s12_star, alpha=0.7, edgecolors='none', marker='s', cmap='seismic', s=s, vmin=-1.5, vmax=1.5)
    ax[1, 2].axis('square')
    # for key, spine in ax[1, 2].spines.items():
    #     if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    #         spine.set_visible(False)
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_xlim([xmin+15, xmax+15])
    ax[1, 2].set_ylim([xmin+15, xmax+15])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 2].set_title(r'$\sigma_{12}$-FEM', fontsize=22)
    # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 2], fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=18)

    # plt.show()
    plt.savefig('./output/stress_comparison_'+str(num).zfill(3)+'.png', dpi=dpi)
    plt.close('all')


def plotResidual(xmin, xmax, ymin, ymax, field=[], s=12, num=0, dpi=100):
    ''' num: Number of time step
    '''
    [x_pred, y_pred, _, u_error, v_error, s11_error, s22_error, s12_error] = field

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.1)

    cf = ax[0, 0].scatter(x_pred, y_pred, c=u_error, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s)
    ax[0, 0].axis('square')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title(r'|$u_{pred}-u_{ref}$|', fontsize=16)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 0].scatter(x_star, y_star, c=s11_error, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s)
    ax[1, 0].axis('square')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 0].set_title(r'|$\sigma_{11}^{pred}-\sigma_{11}^{ref}$|', fontsize=16)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    cf = ax[0, 1].scatter(x_pred, y_pred, c=v_error, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s)
    ax[0, 1].axis('square')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[0, 1].set_title(r'|$v_{pred}-v_{ref}$|', fontsize=16)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 1].scatter(x_star, y_star, c=s22_error, alpha=0.9, edgecolors='none', cmap='seismic', marker='o', s=s)
    ax[1, 1].axis('square')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 1].set_title(r'|$\sigma_{22}^{pred}-\sigma_{22}^{ref}$|', fontsize=16)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    # cf = ax[0, 2].scatter(x_pred, y_pred, c=amp_pred, alpha=0.9, edgecolors='none', cmap=new_cmap, marker='o', s=s, vmin=0, vmax=0.5)
    # ax[0, 2].axis('square')
    # ax[0, 2].set_xticks([])
    # ax[0, 2].set_yticks([])
    # ax[0, 2].set_xlim([xmin, xmax])
    # ax[0, 2].set_ylim([ymin, ymax])
    # # cf.cmap.set_under('whitesmoke')
    # # cf.cmap.set_over('black')
    # ax[0, 2].set_title('Mag.-PINN', fontsize=22)
    # # cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 2], fraction=0.046, pad=0.04)
    # # cbar.ax.tick_params(labelsize=18)

    cf = ax[1, 2].scatter(x_star, y_star, c=s12_error, alpha=0.9, edgecolors='none', cmap=new_cmap, marker='o', s=s)
    ax[1, 2].axis('square')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1, 2].set_title(r'|$\sigma_{12}^{pred}-\sigma_{12}^{ref}$|', fontsize=16)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    # plt.draw()
    plt.savefig('./residual/residual_'+str(num).zfill(3)+'.png', dpi=dpi)
    plt.close('all')


def GenCirclePT(xc, yc, r, N_PT):

    # XY = np.array([-r, -r]) + np.array([2*r, 2*r]) * lhs(2, N_PT)
    # dst = np.array([(xy[0]**2+xy[1]**2)**0.5 for xy in XY])

    # plt.scatter(XY[:,0], XY[:,1])
    # plt.show()

    # xx = XY[dst<=r,0:1] + xc
    # yy = XY[dst<=r,1:2] + yc

    theta = np.linspace(0.0, np.pi*2, N_PT)
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]

    return xx, yy


def DelSrcPT(XYT_c, xc, yc, r):
    dst = np.array([((xyt[0] - xc) ** 2 + (xyt[1] - yc) ** 2) ** 0.5 for xyt in XYT_c])

    return XYT_c[dst>r,:]

def shuffle(XYT_c, SRC, IC, UP):
    # Shuffle along the first dimension
    np.random.shuffle(XYT_c)
    np.random.shuffle(SRC)
    np.random.shuffle(IC)
    np.random.shuffle(UP)

if __name__ == "__main__":

    PI = math.pi
    MAX_T = 14.0              # Pretraining will help convergence (train 7s -> 14s)

    # Domain bounds
    lb = np.array([-15.0, -15.0, 0.0])
    ub = np.array([ 15.0,  15.0, MAX_T])

    # Network configuration, there is only Dirich. boundary (u,v)
    uv_layers = [3] + 6*[140] + [7]
    dist_layers = [3] + 3 * [30] + [5]
    part_layers = [3] + 3 * [20] + [5]

    # Properties of source
    xc_src = 0.0
    yc_src = 0.0
    r_src = 2.0

    # Num of collocation point in x, y, t
    N_f = 120000

    N_t = int(MAX_T*4+1)  # 4 frames per second

    x_dist, y_dist, t_dist = GenDistPt(xmin=-15.0, xmax=15.0, ymin=-15.0, ymax=15.0, tmin=0, tmax=MAX_T, xc=0.0, yc=0.0, r=2.0, num_surf_pt=120, num=21, num_t=21)
    XYT_dist = np.concatenate((x_dist, y_dist, t_dist), 1)
    DIST = GenDist(XYT_dist)

    # Visualize ALL the training points for distance function
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(XYT_dist[:, 0:1], XYT_dist[:, 1:2], XYT_dist[:, 2:3], marker='o', alpha=0.1, s=2, color='blue')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('T axis')
    # plt.show()

    # plt.scatter(DIST[DIST[:,2]==10.0, 0:1], DIST[DIST[:,2]==10.0, 1:2], c=DIST[DIST[:,2]==10.0, 3:4], label='d_u', cmap='rainbow', marker='s', alpha=1)
    # plt.scatter(DIST[DIST[:, 2] == 10.0, 0:1], DIST[DIST[:, 2] == 10.0, 1:2], c=DIST[DIST[:, 2] == 10.0, 4:5], label='d_v', cmap='rainbow', marker='s', alpha=1)
    # plt.scatter(DIST[DIST[:, 2] == 1.0, 0:1], DIST[DIST[:, 2] == 1.0, 1:2], c=DIST[DIST[:, 2] == 1.0, 5:6], label='d_s11', cmap='rainbow', marker='s', alpha=1)
    # plt.scatter(DIST[DIST[:, 2] == 1.0, 0:1], DIST[DIST[:, 2] == 1.0, 1:2], c=DIST[DIST[:, 2] == 1.0, 6:7], label='d_s22',cmap='rainbow', marker='s', alpha=1)
    # plt.scatter(DIST[DIST[:, 2] == 1.0, 0:1], DIST[DIST[:, 2] == 1.0, 1:2], c=DIST[DIST[:, 2] == 1.0, 7:8], label='d_s12', cmap='rainbow', marker='s', alpha=1)
    # plt.colorbar()
    # plt.show()

    # Initial condition point for u, v, ut, vt
    IC = lb + np.array([30.0, 30.0, 0.0]) * lhs(3, 6000)
    IC = DelSrcPT(IC, xc=0, yc=0, r=2.0)

    LW = np.array([-15.0, -15.0, 0.0]) + np.array([30.0, 0.0, MAX_T]) * lhs(3, 7000)

    UP = np.array([-15.0,  15.0, 0.0]) + np.array([30.0, 0.0, MAX_T]) * lhs(3, 7000)

    LF = np.array([-15.0, -15.0, 0.0]) + np.array([0.0, 30.0, MAX_T]) * lhs(3, 7000)

    RT = np.array([ 15.0, -15.0, 0.0]) + np.array([0.0, 30.0, MAX_T]) * lhs(3, 7000)

    FIXED = np.concatenate((LF, RT, LW, UP), 0)

    # Collocation point
    XYT_c = lb + (ub - lb) * lhs(3, N_f)
    XYT_c_ext = np.array([xc_src - r_src -1, yc_src - r_src -1, 0.0]) + np.array([2*(r_src+1), 2*(r_src+1), MAX_T]) * lhs(3, 15000)   # Refinement around source
    XYT_c_ext2 = lb + (ub - lb) * lhs(3, 50000)
    flag = (np.abs(XYT_c_ext2[:,0])>12)|(np.abs(XYT_c_ext2[:,1])>12)
    XYT_c_ext2 = XYT_c_ext2[flag,:]
    XYT_c = np.concatenate((XYT_c, XYT_c_ext, XYT_c_ext2),axis=0)
    XYT_c = DelSrcPT(XYT_c, xc=xc_src, yc=yc_src, r=r_src)

    plt.scatter(XYT_c[:, 0], XYT_c[:, 1], marker='o', alpha=0.2 )
    plt.show()

    # Wave source, Gauss pulse
    tsh = 3.0
    ts = 3.0
    Amp = 1.0
    xx, yy = GenCirclePT(xc=xc_src, yc=yc_src, r=r_src, N_PT=200)   # N_PT=500
    tt = np.concatenate((np.linspace(0, 4, 141), np.linspace(4, MAX_T, 141)),0)
    tt = tt[1:]
    x_SRC, t_SRC = np.meshgrid(xx, tt)
    y_SRC, _     = np.meshgrid(yy, tt)
    x_SRC = x_SRC.flatten()[:, None]
    y_SRC = y_SRC.flatten()[:, None]
    t_SRC = t_SRC.flatten()[:, None]
    # amplitude = Amp*(2*PI**2*(t_SRC-ts)**2/tsh**2-1)*np.exp(-PI**2*(t_SRC-ts)**2/tsh**2)
    amplitude = 0.5 * np.exp(-((t_SRC - 2.0) / 0.5) ** 2)
    u_SRC = amplitude*(x_SRC-xc_src)/r_src
    v_SRC = amplitude*(y_SRC-yc_src)/r_src
    SRC = np.concatenate((x_SRC, y_SRC, t_SRC, u_SRC, v_SRC), 1)

    # plt.scatter(t_SRC, amplitude, marker='o', alpha=0.2 )
    # plt.show()

    # plt.scatter(x_SRC, y_SRC, marker='o', alpha=0.2 )
    # plt.show()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(XYT_c[:,0:1], XYT_c[:,1:2], marker='o', alpha=0.1, s=1)
    ax.scatter(SRC[:, 0:1], SRC[:, 1:2], marker='o', alpha=0.2, color='red', s=2)
    ax.scatter(FIXED[:, 0:1], FIXED[:, 1:2], marker='o', alpha=0.2, color='orange', s=2)
    # ax.scatter(IC[:, 0:1], IC[:, 1:2], marker='o', alpha=0.2)
    plt.show()

    # Visualize ALL the training points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(XYT_c[:,0:1], XYT_c[:,1:2], XYT_c[:,2:3], marker='o', alpha=0.1, s=2, color='blue')
    # ax.scatter(SRC[:, 0:1], SRC[:, 1:2], SRC[:, 2:3], marker='o', alpha=0.3, s=2,color='red')
    # ax.scatter(FIXED[:, 0:1], FIXED[:, 1:2], FIXED[:, 2:3], marker='o', alpha=0.3, s=2, color='blue')
    # ax.scatter(IC[:, 0:1], IC[:, 1:2], IC[:, 2:3], marker='o', alpha=0.2)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('T axis')
    # plt.show()

    with tf.device('/device:GPU:1'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # You can either provide the model or train from scratch
        model = DeepElasticWave(XYT_c, SRC, IC, FIXED, DIST, uv_layers, dist_layers, part_layers, lb, ub, uvDir='uv_NN_14s_float64_new.pickle')

        # Train the distance function and check it
        model.train_bfgs_dist(batch_num=1)
        # u_dist_pred, v_dist_pred, s11_dist_pred, s22_dist_pred, s12_dist_pred = model.predict_D(DIST[DIST[:, 2] == 5.0, 0:1], DIST[DIST[:, 2] == 5.0, 1:2], DIST[DIST[:, 2] == 5.0, 2:3])
        model.count = 0

        # Train the NN for Up(x,y) and check it
        model.train_bfgs_part(batch_num=1)
        # P_u, P_v = model.predict_P(DIST[DIST[:, 2] == 2.5, 0:1], DIST[DIST[:, 2] == 2.5, 1:2], DIST[DIST[:, 2] == 2.5, 2:3])
        model.count = 0

        # Freeze the distance function and particular solution, train the composite networks
        start_time = time.time()
        # loss_f_uv, loss_f_s, loss = model.train(iter=5000, learning_rate=1e-3, batch_num=1)
        model.train_bfgs(batch_num=1)
        # model.train_bfgs(batch_num=1)
        print("--- %s seconds ---" % (time.time() - start_time))

        model.save_NN('uv_NN_14s_float64_new.pickle', TYPE='UV')
        model.save_NN('dist_NN_14s_float64.pickle', TYPE='DIST')
        model.save_NN('part_NN_14s_float64.pickle', TYPE='PART')

        model.getloss()

        # Output result at each time step
        x_star = np.linspace(-15, 15, 201)
        y_star = np.linspace(-15, 15, 201)
        x_star, y_star = np.meshgrid(x_star, y_star)
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        dst = ((x_star-xc_src)**2+(y_star-yc_src)**2)**0.5
        x_star = x_star[dst >= r_src]
        y_star = y_star[dst >= r_src]
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        shutil.rmtree('./output', ignore_errors=True)
        os.makedirs('./output')
        for i in range(0, N_t):
            t_star = np.zeros((x_star.size, 1))
            t_star.fill(i*MAX_T/(N_t-1))
            u_pred, v_pred, s11_pred, s22_pred, s12_pred, e11_pred, e22_pred, e12_pred = model.predict(x_star, y_star, t_star)
            field = [x_star, y_star, t_star, u_pred, v_pred, s11_pred, s22_pred, s12_pred]
            amp_pred = (u_pred**2 + v_pred**2)**0.5
            postProcess(xmin=-15, xmax=15, ymin=-15, ymax=15, s=4, field=field, num=i, dpi=150)


    pass
