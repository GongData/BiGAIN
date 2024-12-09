import tensorflow as tf
import numpy as np

class GAN_model:
    def __init__(self, seq_length, input_dim, hidden_dim, learning_rate):
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # 定義模型的輸入佔位符
        self.X = tf.placeholder(tf.float32, shape=[None, seq_length, input_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, seq_length, input_dim])
        
        # 構建模型
        self.G = self.generator(self.Z)
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G, reuse=True)
        
        # 定義損失函數
        self.D_loss = -tf.reduce_mean(tf.log(self.D_real + 1e-8) + tf.log(1 - self.D_fake + 1e-8))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake + 1e-8))
        
        # 獲取變量
        self.vars_G = [var for var in tf.trainable_variables() if 'generator' in var.name]
        self.vars_D = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        
        # 定義優化器
        self.opt_G = tf.train.AdamOptimizer(learning_rate).minimize(self.G_loss, var_list=self.vars_G)
        self.opt_D = tf.train.AdamOptimizer(learning_rate).minimize(self.D_loss, var_list=self.vars_D)
        
        # 初始化會話
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            # 使用LSTM作為生成器
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, z, dtype=tf.float32)
            
            # 全連接層將hidden_dim映射回input_dim
            outputs = tf.layers.dense(outputs, self.input_dim)
            return outputs

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            # 使用LSTM作為判別器
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
            
            # 全連接層輸出一個標量
            outputs = tf.layers.dense(outputs[:, -1, :], 1, activation=tf.sigmoid)
            return outputs

    def train_step(self, batch_x):
        # 生成隨機噪聲
        batch_z = np.random.normal(0, 1, [batch_x.shape[0], self.seq_length, self.input_dim])
        
        # 訓練判別器
        _, d_loss = self.sess.run([self.opt_D, self.D_loss],
                                 feed_dict={self.X: batch_x, self.Z: batch_z})
        
        # 訓練生成器
        _, g_loss = self.sess.run([self.opt_G, self.G_loss],
                                 feed_dict={self.Z: batch_z})
        
        return d_loss, g_loss
