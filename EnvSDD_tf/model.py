import tensorflow.compat.v1 as tf

class GATE():

    def __init__(self, hidden_dims, alpha=0.8, nonlinear=True, weight_decay=0.0001,
                 fusion_lambda=1.0):
        """
        fusion_lambda \in [0,1]:
          =1.0 -> 完全等价原 STAGATE（只用 α）
          <1.0 -> 与微环境先验 β 并行聚合并凸组合
        """
        self.n_layers = len(hidden_dims) - 1
        self.alpha = alpha
        self.W, self.v, self.prune_v = self.define_weights(hidden_dims)
        self.C = {}
        self.prune_C = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay
        self.fusion_lambda = fusion_lambda

    def __call__(self, A, prune_A, Beta, X):
        # ---------- Encoder ----------
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, prune_A, Beta, H, layer)
            if self.nonlinear and layer != self.n_layers - 1:
                H = tf.nn.elu(H)

        self.H = H  # final embedding

        # ---------- Decoder (mirror with tied weights) ----------
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(A, prune_A, Beta, H, layer)
            if self.nonlinear and layer != 0:
                H = tf.nn.elu(H)
        X_ = H

        # ---------- Loss (same as original STAGATE) ----------
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))
        weight_decay_loss = 0.0
        for layer in range(self.n_layers):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer]),
                                             self.weight_decay, name='weight_loss')
        self.loss = features_loss + weight_decay_loss

        # 保存注意力（与原版一致）
        if self.alpha == 0:
            self.Att_l = self.C
        else:
            self.Att_l = {'C': self.C, 'prune_C': self.prune_C}
        return self.loss, self.H, self.Att_l, X_

    # ---------- internal ----------
    def __encoder(self, A, prune_A, Beta, H, layer):
        """H W -> (λ·SpMM(att_base, ·) + (1-λ)·SpMM(Beta, ·))"""
        H_lin = tf.matmul(H, self.W[layer])
        if layer == self.n_layers - 1:
            return H_lin

        # 原 STAGATE 路径（表达驱动 α）
        self.C[layer] = self.graph_attention_layer(A, H_lin, self.v[layer], layer)
        if self.alpha == 0:
            agg_alpha = tf.sparse_tensor_dense_matmul(self.C[layer], H_lin)
        else:
            self.prune_C[layer] = self.graph_attention_layer(prune_A, H_lin, self.prune_v[layer], layer)
            agg_alpha = (1 - self.alpha) * tf.sparse_tensor_dense_matmul(self.C[layer], H_lin) \
                        + self.alpha * tf.sparse_tensor_dense_matmul(self.prune_C[layer], H_lin)

        # 微环境先验路径（β 已按 dst 归一）
        agg_beta = tf.sparse_tensor_dense_matmul(Beta, H_lin)

        # λ 融合（不显式相加稀疏图，稳定且省显存）
        lam = self.fusion_lambda
        return lam * agg_alpha + (1.0 - lam) * agg_beta

    def __decoder(self, A, prune_A, Beta, H, layer):
        H_lin = tf.matmul(H, self.W[layer], transpose_b=True)
        if layer == 0:
            return H_lin

        if self.alpha == 0:
            agg_alpha = tf.sparse_tensor_dense_matmul(self.C[layer - 1], H_lin)
        else:
            agg_alpha = (1 - self.alpha) * tf.sparse_tensor_dense_matmul(self.C[layer - 1], H_lin) \
                        + self.alpha * tf.sparse_tensor_dense_matmul(self.prune_C[layer - 1], H_lin)

        agg_beta = tf.sparse_tensor_dense_matmul(Beta, H_lin)
        lam = self.fusion_lambda
        return lam * agg_alpha + (1.0 - lam) * agg_beta

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(self.n_layers - 1):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))
            Ws_att[i] = v

        if self.alpha == 0:
            return W, Ws_att, None

        prune_Ws_att = {}
        for i in range(self.n_layers - 1):
            prune_v = {}
            prune_v[0] = tf.get_variable("prune_v%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_v[1] = tf.get_variable("prune_v%s_1" % i, shape=(hidden_dims[i+1], 1))
            prune_Ws_att[i] = prune_v
        return W, Ws_att, prune_Ws_att

    def graph_attention_layer(self, A, M, v, layer):
        # STAGATE-style attention: sigmoid -> sparse_softmax (per-dst)
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0]); f1 = A * f1
            f2 = tf.matmul(M, v[1]); f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)
            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)
            return attentions
