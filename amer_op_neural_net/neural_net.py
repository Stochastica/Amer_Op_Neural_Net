'''
The Neural Network object for the American Option pricing and hedging algorithm
'''

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as KB
import tensorflow.keras.optimizers as KO
import tensorflow.keras.optimizers.schedules as KOS
import collections
from .config import *
from .options import levelset_function, payoff_function


class NeuralNet(tensorflow.keras.Model):
    
    def __init__(self, name="AmerOp", dim_layers=[None, d + 5, d + 5, d + 5, d + 5, d + 5, d + 5, d + 5, 1]):
        """
        Constructor of the NeuralNet class
	
        name: string, name of the neural network graph
        dim_layers: list of integers, dimensions of the hidden layers
        """
        super(NeuralNet, self).__init__(name=name)

        self.additional_features = ("P", "Y")
        self.dim_layers = dim_layers
        self.dim_layers[0] = len(self.additional_features)
        self.dim_layers[-1] = 1
        self.num_layers = len(self.dim_layers)
        self.varmap = {}
        self.n = 0

        self.build_hyperparams()
        #self.build_dataset() #V1
        

        ## Definition and initialization of the trainable variables in the network

        # The input shape argument is just a dummy.
        self.build([
            (None, None, d),    # X
            (None, None, 1),    # Y Prestep
            (None, None, d, 1), # gradY Prestep
        ])
        ## Construction of the training graph
        #self.Y, self.gradY = \
        #self.build_graph(graph_type="train", X=self.train_X, Y_prestep=self.train_Y_prestep,
        #                     gradY_prestep=self.train_gradY_prestep, sharpness=sharpness)
        ## Definition of the loss function and the optimizer
        #self.loss(dA=self.dA, YLabel=self.YLabel, Y=self.Y, gradY=self.gradY)

        ## Construction of the test graph, which shares the same trainable variable of the training graph
        #self.ensemble_Y, self.ensemble_gradY, self.input_mv_init_op = \
        #self.build_graph(graph_type="test", X=self.test_X, Y_prestep=self.test_Y_prestep,
        #                     gradY_prestep=self.test_gradY_prestep, sharpness=sharpness)

        #self.check_graph() #V1
        self.optimizer = KO.Adam()
        #self.optimizer = KO.RMSProp()
        
        # Build the computational graph
        
        sh = (1,1)
        x = tf.zeros(sh + (d,))
        dA = tf.zeros(sh + (d,))
        yLabel = tf.zeros(sh + (1,))
        yPrestep = tf.zeros(sh + (1,))
        gradYPrestep = tf.zeros(sh + (d,1))

        self.bn_rate = tf.cast(0.9, tf.float32)
        
        self.call([x, yPrestep, gradYPrestep], training=True)
        self.call([x, yPrestep, gradYPrestep], training=False)

    def build_hyperparams(self):
        """
        Definition of training hyperparameters
        """
        self.step = tf.Variable(name="step",
                                initial_value=tf.constant_initializer(0)(dtype=tf.int32, shape=[]),
                                trainable=False)
        self.init_rate = tf.Variable(name="init_rate",
                                     initial_value=tf.constant_initializer(0.0)(dtype=tf.float32, shape=[]),
                                     trainable=False)
        self.decay_rate = tf.Variable(name="decay_rate",
                                      initial_value=tf.constant_initializer(1.0)(dtype=tf.float32, shape=[]),
                                      trainable=False)
        self.n_relaxstep = tf.Variable(name="n_relaxstep",
                                       initial_value=tf.constant_initializer(0)(dtype=tf.int32, shape=[]),
                                       trainable=False)
        self.n_decaystep = tf.Variable(name="n_decaystep",
                                       initial_value=tf.constant_initializer(0)(dtype=tf.int32, shape=[]),
                                       trainable=False)
        
        ## self.rate: Global learning rate of the Adam optimizer
        self.rate_sched = KOS.ExponentialDecay(
            self.init_rate,
            decay_steps=self.n_decaystep,
            decay_rate=self.decay_rate,
            staircase=False)
        ## self.bn_rate: Learning rate of the batch normalization
        self.bn_init_rate = 1.0
        self.bn_rate_sched = KOS.ExponentialDecay(
            self.bn_init_rate,
            decay_steps=self.n_decaystep,
            decay_rate=self.decay_rate,
            staircase=False)

    def _rate(self, step):
        return self.rate_sched(tf.clip_by_value(step - self.n_relaxstep, 0, self.n_decaystep))
    def _bn_rate(self, step):
        return self.bn_init_rate \
                * (self.bn_rate_sched(tf.clip_by_value(step, 0, self.n_decaystep)) - self.decay_rate) \
                / (1 - self.decay_rate)
    @property
    def ensemble_b(self):
        return tf.reduce_mean(self.b)

    #def build_dataset(self):
    #    '''
    #    Define input tensor placeholders, training input tensors and test input tensors
    #    '''
    #    self.ph_X = tf.placeholder(dtype=tf.float32, shape=[None, None, d], name="X")
    #    self.ph_dA = tf.placeholder(dtype=tf.float32, shape=[None, None, d], name="dA")
    #    self.ph_YLabel = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name="YLabel")
    #    self.ph_Y_prestep = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name="Y_prestep")
    #    self.ph_gradY_prestep = tf.placeholder(dtype=tf.float32, shape=[None, None, d, 1], name="gradY_prestep")
#
    #    train_dataset = tf.data.Dataset.from_tensor_slices(
    #        (self.ph_X, self.ph_dA, self.ph_YLabel, self.ph_Y_prestep, self.ph_gradY_prestep)).batch(batch_size)
    #    train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    #    self.train_init_op = train_iter.make_initializer(train_dataset)
    #    self.train_X, self.dA, self.YLabel, self.train_Y_prestep, self.train_gradY_prestep = train_iter.get_next()
#
    #    test_dataset = tf.data.Dataset.from_tensor_slices(
    #        (self.ph_X, self.ph_Y_prestep, self.ph_gradY_prestep)).batch(tf.cast(tf.shape(self.ph_X)[0], tf.int64))
    #    test_iter = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    #    self.test_init_op = test_iter.make_initializer(test_dataset)
    #    self.test_X, self.test_Y_prestep, self.test_gradY_prestep = test_iter.get_next()

    def _batch_norm_layer_variables(self, dim):
        '''
        Define and initialize the trainable variables of a batch normalization layer, which will become part of the neural network trainable variables

        batch_norm_name: string, the name of the batch normalization layer
        dim: int, dimension of the batch normalization layer (= dimension of the hidden layer)
        '''
        return {
            'mv_mean': tf.Variable(name="mv_mean",
                initial_value=tf.constant_initializer(0.0)(dtype=tf.float32, shape=[1, num_channels, dim]),
                trainable=False),
            'mv_var': tf.Variable(name="mv_var",
                initial_value=tf.constant_initializer(1.0)(dtype=tf.float32, shape=[1, num_channels, dim]),
                trainable=False),
            'beta': tf.Variable(name="beta",
                initial_value=tf.constant_initializer(0.0)(dtype=tf.float32, shape=[1, num_channels, dim]),
                trainable=True),
            'gamma': tf.Variable(name="gamma",
                initial_value=tf.constant_initializer(1.0)(dtype=tf.float32, shape=[1, num_channels, dim]),
                trainable=True),
        }

    @staticmethod
    @tf.function
    def _assign(x, y):
        return x.assign(tf.broadcast_to(y, x.shape))

    @tf.function
    def _batch_norm_layer(self, Z, input_mv=False, tol=1e-8,
            mv_mean=None, mv_var=None, beta=None, gamma=None):
        '''
        Define a batch normalization layer, which will become part of the neural network

        batch_norm_name: string, the name of the batch normalization layer
        Z: hidden layer value
        input_mv: boolean, whether a batch normalization layer is an input normalization (or a hidden layer normalization)
        '''
        #mv_mean = layerobj['mv_mean']
        #mv_var = layerobj['mv_var']
        #beta = layerobj['beta']
        #gamma = layerobj['gamma']
        batch_mean, batch_var = tf.nn.moments(Z, axes=list(range(len(Z.get_shape())-2)), keepdims=True)
        if input_mv:
            input_mv_init_op = [
                self._assign(mv_mean, batch_mean),
                self._assign(mv_var, batch_var),
            ]
            Z = tf.nn.batch_normalization(Z, mv_mean, mv_var, beta, gamma, tol)
            return input_mv_init_op, Z
        else:
            train_mv_mean = self._assign(mv_mean, mv_mean * (1.0 - self.bn_rate) + batch_mean * self.bn_rate)
            train_mv_var  = self._assign(mv_var,  mv_var  * (1.0 - self.bn_rate) + batch_var  * self.bn_rate)
            with tf.control_dependencies([train_mv_mean, train_mv_var]):
                Z = tf.nn.batch_normalization(Z, mv_mean, mv_var, beta, gamma, tol)
            return Z

    @tf.function
    def __tensor_chainrule(self, gradZ1, gradZ2):
        return tf.expand_dims(gradZ1, axis=-2) * gradZ2

    @tf.function
    def __tensor_contract(self, X, W):
        dim0 = tf.shape(X)[0]
        Wtile = tf.tile(tf.expand_dims(W, axis=0), (dim0, 1, 1, 1))
        return tf.squeeze(tf.matmul(tf.expand_dims(X, axis=-2), Wtile), axis=-2)

    def build(self, input_shapes):
        """
        Define and initialize all the trainable variables of each network
        """
        #super(NeuralNet, self).build(input_shapes)
        stddev = 1.0
        #### Input layers
        layer = 0
        layerobj = {}
        layerobj["batchX"] = self._batch_norm_layer_variables(d)
        layerobj["batchZ"] = self._batch_norm_layer_variables(self.dim_layers[0])
        self.varmap["layer" + str(layer)] = layerobj

        #### Hidden layers
        for layer in range(1, self.num_layers - 1):
            layerobj = {}
                
            if layer == 1:
                lim = stddev / math.sqrt(d + self.dim_layers[1])
                layerobj["WX"] = tf.Variable(name="WX",
                                    initial_value=tf.random_uniform_initializer(-lim,lim)(dtype=tf.float32,
                                     shape=[num_channels, d, self.dim_layers[1]]),
                                     trainable=True)
                lim = stddev / math.sqrt(self.dim_layers[0] + self.dim_layers[1])
                layerobj["WZ"] = tf.Variable(name="WZ",
                                    initial_value=tf.random_uniform_initializer(-lim,lim)(dtype=tf.float32,
                                     shape=[num_channels, self.dim_layers[0], self.dim_layers[1]]),
                                     trainable=True)

            if layer >= 2:
                lim = stddev / math.sqrt(self.dim_layers[layer - 1] + self.dim_layers[layer])
                layerobj["WZ"] = tf.Variable(name="WZ",
                                    initial_value=tf.random_uniform_initializer(-lim,lim)(dtype=tf.float32,
                                     shape=[num_channels, self.dim_layers[layer - 1], self.dim_layers[layer]]),
                                     trainable=True)

            layerobj["batchZ"] = self._batch_norm_layer_variables(self.dim_layers[layer])
            self.varmap["layer" + str(layer)] = layerobj

        #### Output layer
        self.delta = tf.Variable(name="delta",
                initial_value=tf.constant_initializer(1.0)(dtype=tf.float32, shape=[]),
                trainable=False)
        self.alpha = tf.Variable(name="alpha",
                initial_value=tf.constant_initializer(1.0)(dtype=tf.float32, shape=[1, num_channels, 1]),
                trainable=True)
        self.ensemble_alpha = tf.reduce_mean(self.alpha)

        layer = self.num_layers - 1
        layerobj = {}
        lim = stddev / math.sqrt(self.dim_layers[layer - 1] + 1)
        self.WZ = tf.Variable(name="WZ",
                initial_value=tf.random_uniform_initializer(-lim,lim)(dtype=tf.float32,
                                  shape=[num_channels, self.dim_layers[layer - 1], 1]),
                trainable=True)
        self.b = tf.Variable(name="b",
                initial_value=tf.constant_initializer(0.0)(dtype=tf.float32, shape=[1, num_channels, 1]),
                trainable=True)
        #self.ensemble_b = tf.reduce_mean(self.b)
        self.varmap["layer" + str(layer)] = layerobj

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=True):
        """
        Construct the network
        
        Inputs: X, Y_prestep, gradY_prestep, [sharpness]
    
        self.call(x, dA, yLabel, yPrestep, gradYPrestep)
        """
        X, Y_prestep, gradY_prestep = inputs

        #### Input layers
        layer = 0
        layerobj = self.varmap["layer" + str(layer)]
        L = levelset_function(X, sharpness=sharpness)
        input_mv_init_op, X_normal = self._batch_norm_layer(X, input_mv=True, **layerobj['batchX'])
        Z = tf.concat([Y_prestep, L], axis=-1)
        input_mv_init_op_Z, Z_normal = self._batch_norm_layer(Z, input_mv=True, **layerobj['batchZ'])
        input_mv_init_op += input_mv_init_op_Z
        #input_mv_init_op = tf.group(input_mv_init_op)

        #### Hidden layers
        for layer in range(1, self.num_layers - 1):
            layerobj = self.varmap["layer" + str(layer)]
            
            ## Linear transformation
            if layer == 1:
                WX = layerobj["WX"]
                WZ = layerobj["WZ"]
                Z = self.__tensor_contract(X_normal, WX) + self.__tensor_contract(Z_normal, WZ)

            if layer >= 2:
                WZ = layerobj["WZ"]
                Z = self.__tensor_contract(Z, WZ)

            ## Batch normalization
            Z = self._batch_norm_layer(Z, **layerobj['batchZ'])

            ## Nonlinear activation
            Z = tf.nn.relu(Z)

        #### Output layer
        layer = self.num_layers - 1

        Z = self.__tensor_contract(Z, self.WZ) + self.b
        Y = self.alpha * (Y_prestep + self.delta * dt * Z)

        gradY = tf.expand_dims(tf.gradients(Y, X)[0], axis=-1) \
                + self.__tensor_chainrule(tf.gradients(Y, Y_prestep)[0], gradY_prestep)
        
        if training:
            return Y, gradY
        else:
            ensemble_Y = tf.reduce_mean(Y, axis=1, keepdims=True)
            ensemble_gradY = 1 / num_channels * tf.reduce_mean(gradY, axis=1, keepdims=True)
            return ensemble_Y, ensemble_gradY #, input_mv_init_op

    @tf.function
    def loss(self, dA, YLabel, Y, gradY):
        dA = tf.expand_dims(dA, axis=-1)
        Res = YLabel - Y - math.exp(-r * dt) * tf.reduce_sum(dA * gradY, axis=-2)
        return tf.reduce_mean(Res ** 2)


    def check_graph(self):
        '''
        Print out the graph for examination
        '''
        print("\n Trainable variables in the main neural network ", self.name, ": ")
        for Var in self.trainable_variables:
            print("   ", Var)
        #print("\n Global variables in the main neural network ", self.name, ": ")
        #for Var in tf.global_variables():
        #    print("   ", Var)

    def fit(self, Xn, dAn, YLabeln, Y_prestepn, gradY_prestepn, n_totalstep, n_unitstep):
        '''
        Train a neural network using the previously defined optimizer
        '''
        assert self.n < N
        print("\n **** ", self.name, ", Adam Optimization : ")
        #### Data input 
        #sess.run(self.train_init_op, feed_dict={self.ph_X: Xn, self.ph_dA: dAn, self.ph_YLabel: YLabeln,
        #                                        self.ph_Y_prestep: Y_prestepn, self.ph_gradY_prestep: gradY_prestepn})

        #### Main Optimization
        for step in range(n_totalstep):
            rate = self._rate(step)
            self.bn_rate = self._bn_rate(step)

            KB.set_value(self.optimizer.learning_rate, tf.cast(rate, tf.float32))
            with tf.GradientTape() as tape:
                Y, gradY = self([Xn, Y_prestepn, gradY_prestepn], training=True)

                loss = self.loss(dAn, YLabeln, Y, gradY)
            grad = tape.gradient(loss, self.trainable_weights)
            assert len(grad) == len(self.trainable_weights)

            def cond_clip(g, v):
                if "/alpha:0" in v.name:
                    return (g, v)
                else:
                    return (tf.clip_by_value(g, -10., 10.), v)
            grad = [cond_clip(g, v) for g,v in zip(grad, self.trainable_weights)]

            self.optimizer.apply_gradients(grad)

            if step < n_unitstep or step % n_unitstep == 0 or step == n_totalstep - 1:
                print(" step : ", "{:4d}".format(step),
                      ",  learning rate: ", "{0:.6f}".format(float(rate)),
                      ",  batch learning rate: ", "{0:.6f}".format(float(self.bn_rate)),
                      ",  b : ", "{0:.6f}".format(float(self.ensemble_b)),
                      ",  gradY : ", "{0:.6f}".format(float(tf.reduce_mean(gradY))),
                      ",  regloss : ", "{0:.6f}".format(float(loss * 10000)))

    def train(self, n, AmerOp, n_totalstep, n_unitstep, n_relaxstep, n_decaystep,
              init_rate, decay_rate):
        '''
        Implement the proposed American option algorithm for the n-th timestep
        '''
        print("\n **** ", self.name, ", Training : ")
        self.n = n
        self.delta.assign(updaten - self.n % updaten)
        self.init_rate.assign(init_rate)
        self.decay_rate.assign(decay_rate)
        self.n_relaxstep.assign(n_relaxstep)
        self.n_decaystep.assign(n_decaystep)

        def customer_reshape(Xn):
            Xn = tf.cast(Xn, tf.float32)
            if Xn.ndim == 2:
                return tf.reshape(Xn, (len(Xn) // num_channels, num_channels, Xn.shape[1]))
            if Xn.ndim == 3:
                return tf.reshape(Xn, (len(Xn) // num_channels, num_channels, Xn.shape[1], Xn.shape[2]))

        #### Data input
        np.random.shuffle(simulation_index)
        Xn = customer_reshape(AmerOp.X.values[simulation_index, self.n])
        dAn = customer_reshape(AmerOp.X.values[simulation_index, self.n + 1]) - np.exp(mu * dt) * Xn
        YLabeln = customer_reshape(AmerOp.YLabel.values[simulation_index, self.n])
        Y_prestepn = customer_reshape(AmerOp.Y.values[simulation_index, self.n])
        gradY_prestepn = customer_reshape(AmerOp.gradY.values[simulation_index, self.n])

        #sess.run(self.test_init_op, feed_dict={self.ph_X: Xn, self.ph_Y_prestep: Y_prestepn,
        #                                       self.ph_gradY_prestep: gradY_prestepn})
        #sess.run(self.input_mv_init_op)

        #### Main Optimization
        t0 = time.time()
        self.fit(Xn, dAn, YLabeln, Y_prestepn, gradY_prestepn, n_totalstep, n_unitstep)
        print("\n   time: ", "{0:.2f}".format(time.time() - t0))

    def predict_Y_gradY_timestep(self, AmerOp, m):
        assert self.n < N
        print("\n Ensemble Average : ", str(m))
        t0 = time.time()
        ensemble_Y, ensemble_gradY = self([
            AmerOp.X.values[:, [m]],
            AmerOp.Y.values[:, [m]],
            AmerOp.gradY.values[:, [m]],
        ], training=False)
        AmerOp.Y.values[:, [m]] = ensemble_Y
        AmerOp.gradY.values[:, [m]] = ensemble_gradY
        # assert ensemble_Y.shape == (simulation_size, 1, 1)
        # assert ensemble_gradY.shape == (simulation_size, 1, d, 1)
        print(" time: " + "{0:.2f}".format(time.time() - t0))

    def predict(self, AmerOp):
        t0 = time.time()
        if self.n % updaten == 0:
            for m in range(1, self.n + 1):
                self.predict_Y_gradY_timestep(AmerOp=AmerOp, m=m)
        else:
            self.predict_Y_gradY_timestep(AmerOp=AmerOp, m=self.n)

