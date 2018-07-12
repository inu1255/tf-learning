import tensorflow as tf
import numpy as np
from lib import utils
from lib.layers import shufflenet_unit, conv2d, max_pool_2d, avg_pool_2d, dense, flatten

class NeuralNetwork(object):
    def __init__(self, shape, save_path='model/default.ckpt'):
        # 传入变量
        self.inputs = tf.placeholder(tf.float32, shape=[None, *shape])
        self.outputs = self.inputs
        self.is_train = tf.placeholder(tf.bool)
        self.save_path = save_path
        # 私有变量
        self.layer_count = 0
        self.regular_weights = []
        self.regular_biases = []
        # 缓存变量
        self.saver = None
        self.session = None

    def layer_name(self, layer, name):
        self.layer_count += 1
        if name is None:
            return '%s%d'%(layer, self.layer_count)
        return name

    def fc(self, out_num_nodes, activation='relu', dropout=False, name=None):
        name = self.layer_name('fc', name)
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([int(self.outputs.shape[1]), out_num_nodes], stddev=0.1))
            self.regular_weights.append(weights)
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes]))
            self.regular_biases.append(biases)
        with tf.name_scope(name+'model'):
            self.outputs = tf.matmul(self.outputs, weights) + biases
            if not activation:
                pass
            elif activation=='relu':
                self.outputs = tf.nn.relu(self.outputs)
            else:
                raise Exception('Activation Func not support "%s" now'%activation)
            if dropout:
                self.outputs = tf.cond(self.is_train, lambda: tf.nn.dropout(self.outputs, 0.9, seed=1255), lambda: self.outputs) 

    def apply_regularization(self, _lambda=5e-4):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.regular_weights, self.regular_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        # 1e5
        return _lambda * regularization
    
    def accuracy(self, predictions, labels):
        '''
        计算预测的正确率
        '''
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy
    
    def confusion_matrix(self, predictions, labels):
        '''
        计算预测的召回率
        '''
        from sklearn.metrics import confusion_matrix
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions)
        return cm

    def get_session(self, force=False):
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables())
        if force or self.session is None:
            self.session = tf.Session(graph=tf.get_default_graph())
            self.session.run(tf.global_variables_initializer())
            try:
                self.saver.restore(self.session, self.save_path)
                print('读取模型:%s'%self.save_path)
            except Exception as e:
                pass
        return self.session

    def train(self, train_samples, train_labels, optimizer=None, *, iteration_steps, chunkSize):
        shape = train_labels.shape
        tf_train_labels = tf.placeholder(tf.float32, shape=(chunkSize, *shape[1:]), name='tf_train_labels')
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=tf_train_labels))
            loss += self.apply_regularization(_lambda=5e-4)
        if optimizer is None:
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(
                learning_rate=0.001,
                global_step=global_step*chunkSize,
                decay_steps=100,
                decay_rate=0.99,
                staircase=True
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(loss)
        with tf.name_scope('train'):
            train_prediction = tf.nn.softmax(self.outputs, name='train_prediction')
        session = self.get_session(force=True)
        for i, samples, labels in utils.train_data_iterator(train_samples, train_labels, iteration_steps=iteration_steps, chunkSize=chunkSize):
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction],
                feed_dict={self.inputs: samples, tf_train_labels: labels, self.is_train: True}
            )
            # labels is True Labels
            if i % 50 == 0:
                accuracy = self.accuracy(predictions, labels)
                print('Minibatch loss at step %d: %f' % (i, l))
                print('Minibatch accuracy: %.1f%%' % accuracy)
        ###

        # 检查要存放的路径值否存在。这里假定只有一层路径。
        import os
        if os.path.isdir(os.path.dirname(self.save_path)):
            save_path = self.saver.save(session, self.save_path)
            print("Model saved in file: %s" % save_path)
        else:
            os.makedirs(os.path.dirname(self.save_path))
            save_path = self.saver.save(session, self.save_path)
            print("Model saved in file: %s" % save_path)

    def test(self, test_samples, test_labels, *, chunkSize):
        with tf.name_scope('test'):
            test_prediction = tf.nn.softmax(self.outputs, name='test_prediction')

        session = self.get_session()
        ### 测试
        accuracies = []
        confusionMatrices = []
        for i, samples, labels in utils.test_data_iterator(test_samples, test_labels, chunkSize=chunkSize):
            result= session.run(test_prediction, feed_dict={self.inputs: samples, self.is_train: False})
            accuracy = self.accuracy(result, labels)
            cm = self.confusion_matrix(result, labels)
            accuracies.append(accuracy)
            confusionMatrices.append(cm)
            print('Test Accuracy: %.1f%%' % accuracy)
        print(' Average  Accuracy:', np.average(accuracies))
        print('Standard Deviation:', np.std(accuracies))
        self.print_confusion_matrix(np.add.reduce(confusionMatrices))
        ###

    def predict(self, input):
        prediction = tf.nn.softmax(self.outputs, name='prediction')
        session = self.get_session()
        return session.run(prediction, feed_dict={self.inputs: [input], self.is_train: False})

    def print_confusion_matrix(self, confusionMatrix):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            print('真实值%d: %s %.3f%%'%(i, line, 100 * line[i] / np.sum(line)))
        a = 0
        total = np.sum(confusionMatrix)
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / total)
            print('预测为%d: %.3f%%'%(i, 100 * column[i] / np.sum(column)), )
        print('\n', total, a)

class CNN(NeuralNetwork):
    def __init__(self, *args, **kargs):
        NeuralNetwork.__init__(self, *args, **kargs)
        self.reshaped = False

    def conv2d(self, patch_size, out_depth=32, strides=1, padding='SAME', activation='relu', pooling=False, pooling_scale=2, pooling_stride=2, pooling_padding=None, name=None):
        name = self.layer_name('conv2d', name)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([patch_size[0], patch_size[1], int(self.outputs.shape[3]), out_depth], stddev=0.1), name=name+'_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[32]), name=name+'_biases')
        with tf.name_scope(name+'_model'):
            with tf.name_scope('convolution'):
                # default 1,1,1,1 stride and SAME padding
                self.outputs = tf.nn.conv2d(self.outputs, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding)
                self.outputs = self.outputs + biases
                if not activation:
                    pass
                elif activation=='relu':
                    self.outputs = tf.nn.relu(self.outputs)
                else:
                    raise Exception('Activation Func not support "%s" now'%activation)
                if pooling:
                    if isinstance(pooling_scale, int):
                        pooling_scale = (pooling_scale, pooling_scale)
                    if isinstance(pooling_stride, int):
                        pooling_stride = (pooling_stride, pooling_stride)
                    if pooling_padding is None:
                        pooling_padding = padding
                    self.outputs = tf.nn.max_pool(self.outputs, ksize=[1, pooling_scale[0], pooling_scale[1], 1], strides=[1, pooling_stride[0], pooling_stride[0], 1], padding=pooling_padding)

    def fc(self, out_num_nodes, activation='relu', dropout=False, name=None):
        if not self.reshaped:
            self.outputs = tf.layers.flatten(self.outputs)
            self.reshaped = True
        return NeuralNetwork.fc(self, out_num_nodes, activation=activation, dropout=dropout, name=name)

class ShuffleNet(CNN):
    MEAN = [103.94, 116.78, 123.68]
    NORMALIZER = 0.017
    OUTPUT_CHANNELS = {
        '1': [144, 288, 576], 
        '2': [200, 400, 800], 
        '3': [240, 480, 960], 
        '4': [272, 544, 1088],
        '8': [384, 768, 1536], 
        'conv1': 24
    }

    def __init__(self, *args, **kargs):
        CNN.__init__(self, *args, **kargs)

    def __stage(self,output_channels, x, stage=2, repeat=3):
        num_groups = 3
        l2_strength = 4e-5
        if 2 <= stage <= 4:
            stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None,
                                          num_groups=num_groups,
                                          group_conv_bottleneck=not (stage == 2),
                                          num_filters=
                                          output_channels[str(num_groups)][
                                              stage - 2],
                                          stride=(2, 2),
                                          fusion='concat', l2_strength=l2_strength,
                                          bias=0.0,
                                          batchnorm_enabled=True,
                                          is_training=True)
            for i in range(1, repeat + 1):
                stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i),
                                              x=stage_layer, w=None,
                                              num_groups=num_groups,
                                              group_conv_bottleneck=True,
                                              num_filters=output_channels[
                                                  str(num_groups)][stage - 2],
                                              stride=(1, 1),
                                              fusion='add',
                                              l2_strength=l2_strength,
                                              bias=0.0,
                                              batchnorm_enabled=True,
                                              is_training=True)
            return stage_layer
        else:
            raise ValueError("Stage should be from 2 -> 4")

    def suffle(self, output_channels=None):
        l2_strength = 4e-5
        if output_channels is None:
            output_channels = ShuffleNet.OUTPUT_CHANNELS
        with tf.name_scope('Preprocessing'):
            red, green, blue = tf.split(self.outputs, num_or_size_splits=3, axis=3)
            preprocessed_input = tf.concat([
                tf.subtract(blue, ShuffleNet.MEAN[0]) * ShuffleNet.NORMALIZER,
                tf.subtract(green, ShuffleNet.MEAN[1]) * ShuffleNet.NORMALIZER,
                tf.subtract(red, ShuffleNet.MEAN[2]) * ShuffleNet.NORMALIZER,
            ], 3)
        x_padded = tf.pad(preprocessed_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        conv1 = conv2d('conv1', x=x_padded, w=None, num_filters=output_channels['conv1'], kernel_size=(3, 3),
                       stride=(2, 2), l2_strength=l2_strength, bias=0.0,
                       batchnorm_enabled=True, is_training=True,
                       activation=tf.nn.relu, padding='VALID')
        padded = tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
        max_pool = max_pool_2d(padded, size=(3, 3), stride=(2, 2), name='max_pool')
        stage2 = self.__stage(output_channels, max_pool, stage=2, repeat=3)
        stage3 = self.__stage(output_channels, stage2, stage=3, repeat=7)
        stage4 = self.__stage(output_channels, stage3, stage=4, repeat=3)
        # global_pool = avg_pool_2d(stage4, size=(7, 7), stride=(1, 1), name='global_pool', padding='VALID')

        logits_unflattened = conv2d('fc', stage4, w=None, num_filters=10,
                                    kernel_size=(1, 1),
                                    l2_strength=l2_strength,
                                    bias=0.0,
                                    is_training=True)
        self.outputs = flatten(logits_unflattened)


def simple_conv(shape, save_path='model/simple_conv/default.ckpt'):
    net = CNN(shape, save_path=save_path)
    net.conv2d(3, name='conv1')
    net.conv2d(3, pooling=True, name='conv2')
    net.conv2d(3, name='conv3')
    net.conv2d(3, pooling=True, name='conv4')

    net.fc(128, dropout=True, name='fc1')
    net.fc(10, name='fc2')

    return net

def shufflenet(shape, save_path='model/shufflenet/default.ckpt'):
    net = ShuffleNet(shape, save_path=save_path)
    net.suffle()
    return net