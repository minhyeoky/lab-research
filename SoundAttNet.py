import tensorflow as tf
import numpy as np

from env import GPU_INDEX, ATT
from LatentEncoder import LatentEncoder
from LatentDecoder import LatentDecoder


class SoundAttNet():

    def __init__(self, eta=1e-3):
        self.eta = eta

        self.X = []
        self.att = []
        self.reconstructed = None
        self.loss = None
        self.update_op = None

        self.graph = None
        self.sess = None

        self.summ_op = None
        self.global_step = 1
        
    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def build(self):
        print("Building SoundAttNet")

        self.graph = tf.Graph()

        with self.graph.as_default():
            params = self._init_parameters()

            weights_penalty = \
                    tf.reduce_mean(params["latent_encoder_sex_mean"]**2) + \
                    tf.reduce_mean(params["latent_encoder_sex_sd"]**2) + \
                    tf.reduce_mean(params["latent_encoder_langNat_mean"]**2) + \
                    tf.reduce_mean(params["latent_encoder_langNat_sd"]**2) + \
                    tf.reduce_mean(params["latent_encoder_levKor_mean"]**2) + \
                    tf.reduce_mean(params["latent_encoder_levKor_sd"]**2)

            loss_list = []
            rec_list = []
            grad_var_pair_list = []

            optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)

            for i in range(len(GPU_INDEX)):
                with tf.device(f"/gpu:{i}"):
                    with tf.name_scope(f"sound-att-net-{i+1}"):
                        X, att = self._init_placeholder()
                        self.X.append(X)
                        self.att.append(att)
                        
                        self._summary(tf.summary.audio, f"X_{i}", X, 44100)

                        encoder = LatentEncoder()
                        sex, langNat, levKor = encoder.build(X, att, params)

                        sex = self._sampling_given_mean_sd(*sex)
                        langNat = self._sampling_given_mean_sd(*langNat)
                        levKor = self._sampling_given_mean_sd(*levKor)

                        sampled = sex + langNat + levKor

                        decoder = LatentDecoder()
                        reconstructed, att_pred = decoder.build(sampled, params)
                        
                        self._summary(tf.summary.audio, f"reconstructed_{i}", reconstructed, 44100)

                        rec_list.append(reconstructed)

                        loss_rec = tf.reduce_mean((X - reconstructed)**2)
                        loss_att = tf.reduce_mean((att - att_pred)**2)

                        loss = 1.5*loss_rec + 1.0*loss_att + 0.5*weights_penalty
                        loss_list.append(loss)

                        grad_var_pair = optimizer.compute_gradients(loss)
                        grad_var_pair_list.append(grad_var_pair)

                self.loss = tf.reduce_mean(loss_list)
                self.reconstructed = tf.concat(rec_list, axis=0)
                
                self._summary(tf.summary.scalar, "loss", self.loss)

                grad_var_pair = self._average_gradients(grad_var_pair_list)
                self.update_op = optimizer.apply_gradients(grad_var_pair)

                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
                
                summ_writer = tf.summary.FileWriter("./logdir", graph=self.graph)
                self.summ_op = tf.summary.merge_all()

                self.saver = tf.train.Saver()

        print("SoundAttNet was built.")

    def save(self, path):
        with self.graph.as_default():
            self.saver.save(self.sess, path)
            print("SoundAttNet was saved.")

    def load(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)
            print("SoundAttNet was loaded.")

    def step(self, X, att):
        with self.graph.as_default():
            feed_dict = self._map_data_to_tensor(X, att)
            _, loss, _ = self.sess.run([self.update_op, self.loss, self.summ_op], feed_dict=feed_dict)

        return loss

    def _map_data_to_tensor(self, X, att):
        n = X.shape[0]
        size = n // len(GPU_INDEX)

        feed_dict = dict()

        for i in range(len(GPU_INDEX)):
            start = i * size
            end = (i+1)*size

            if i == len(GPU_INDEX)-1:
                end = n

            feed_dict[self.X[i]] = X[start:end]
            feed_dict[self.att[i]] = att[start:end]

        return feed_dict

    def _sampling_given_mean_sd(self, mean, sd):
        z = mean + sd * tf.random.normal(tf.shape(sd))
        return z

    def _average_gradients(self, grad_var_pair_list):
        new_grad_var_pair = []

        for grads_vars in zip(*grad_var_pair_list):
            grad = 0.0
            var = None

            for _grad, _var in grads_vars:
                var = _var
                if _grad is not None:
                    grad += _grad

            grad /= len(GPU_INDEX)

            new_grad_var_pair.append((grad, var))

        return new_grad_var_pair

    def _init_placeholder(self):
        with tf.name_scope("in"):
            X = tf.placeholder(tf.float32, shape=(None, 2**16, 1), name="X")
            att = tf.placeholder(tf.float32, shape=(None, 3), name="att")

        return X, att

    def _init_parameters(self):
        params = dict()

        with tf.name_scope("params"):
            params["latent_encoder_W1"] = tf.Variable(tf.random.normal((7, 1+len(ATT), 8)), dtype=tf.float32, name="latent_encoder_W1")
            params["latent_encoder_b1"] = tf.Variable(tf.random.normal((1, 1, 8)), dtype=tf.float32, name="latent_encoder_b1")

            params["latent_encoder_W2"] = tf.Variable(tf.random.normal((7, 8+len(ATT), 16)), dtype=tf.float32, name="latent_encoder_W2")
            params["latent_encoder_b2"] = tf.Variable(tf.random.normal((1, 1, 16)), dtype=tf.float32, name="latent_encoder_b2")

            params["latent_encoder_W3"] = tf.Variable(tf.random.normal((7, 16+len(ATT), 32)), dtype=tf.float32, name="latent_encoder_W3")
            params["latent_encoder_b3"] = tf.Variable(tf.random.normal((1, 1, 32)), dtype=tf.float32, name="latent_encoder_b3")

            params["latent_encoder_W4"] = tf.Variable(tf.random.normal((7, 32+len(ATT), 64)), dtype=tf.float32, name="latent_encoder_W4")
            params["latent_encoder_b4"] = tf.Variable(tf.random.normal((1, 1, 64)), dtype=tf.float32, name="latent_encoder_b4")

            params["latent_encoder_sex_mean"] = tf.Variable(tf.random.normal((7, 64, 64)), dtype=tf.float32, name="latent_encoder_sex_mean")
            params["latent_encoder_sex_sd"] = tf.Variable(tf.random.normal((7, 64, 64)), dtype=tf.float32, name="latent_encoder_sex_sd")

            params["latent_encoder_langNat_mean"] = tf.Variable(tf.random.normal((7, 64, 64)), dtype=tf.float32, name="latent_encoder_langNat_mean")
            params["latent_encoder_langNat_sd"] = tf.Variable(tf.random.normal((7, 64, 64)), dtype=tf.float32, name="latent_encoder_langNat_sd")

            params["latent_encoder_levKor_mean"] = tf.Variable(tf.random.normal((7, 64, 64)), dtype=tf.float32, name="latent_encoder_levKor_mean")
            params["latent_encoder_levKor_sd"] = tf.Variable(tf.random.normal((7, 64, 64)), dtype=tf.float32, name="latent_encoder_levKor_sd")

            params["latent_decoder_W1"] = tf.Variable(tf.random.normal((7, 16, 64)), dtype=tf.float32, name="latent_decoder_W1")
            params["latent_decoder_b1"] = tf.Variable(tf.random.normal((1, 1, 16)), dtype=tf.float32, name="latent_decoder_b1")

            params["latent_decoder_W2"] = tf.Variable(tf.random.normal((7, 1, 16)), dtype=tf.float32, name="latent_decoder_W2")
            params["latent_decoder_b2"] = tf.Variable(tf.random.normal((1, 1, 1)), dtype=tf.float32, name="latent_decoder_b2")

            params["disc_W1"] = tf.Variable(tf.random.normal((2**16, 1024)), dtype=tf.float32, name="disc_W1")
            params["disc_b1"] = tf.Variable(tf.random.normal((1, 1024)), dtype=tf.float32, name="disc_b1")

            params["disc_W2"] = tf.Variable(tf.random.normal((1024, 64)), dtype=tf.float32, name="disc_W2")
            params["disc_b2"] = tf.Variable(tf.random.normal((1, 64)), dtype=tf.float32, name="disc_b2")

            params["disc_W3"] = tf.Variable(tf.random.normal((64, len(ATT))), dtype=tf.float32, name="disc_W3")
            params["disc_b3"] = tf.Variable(tf.random.normal((1, len(ATT))), dtype=tf.float32, name="disc_b3")

        return params

    def _summary(self, summ_fn, *args):
        with tf.device("/cpu:0"):
            summ_fn(*args)
