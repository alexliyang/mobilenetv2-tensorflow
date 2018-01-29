import tensorflow as tf
import tensorflow.contrib.layers as tflayer


class MoblieNetV2(object):
    def __init__(self, is_training: bool):
        self.is_training = is_training

    def construct_network(self, input_shape, output_num, weigth_decay=0.99, input_tensor=None):
        """
        contrust a moblienet v2 model
        :param input_shape:
        :param output_num:
        :param weigth_decay:
        :param input_tensor:
        :return:the prediction tensor with 2-D
        """
        if input_tensor is not None:
            x_input = input_tensor
        else:
            x_input = tf.placeholder(dtype=tf.float32,
                                     shape=input_shape,
                                     name='network_input')
        x = MoblieNetV2.conv_block(inputs=x_input,
                                   output_num=32,
                                   expand_alpha=1,
                                   stride=[2, 2],
                                   the_name='conv1')
        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=1,
                                                out_channels=16,
                                                repeats=1,
                                                stride=1,
                                                weight_decay=weigth_decay,
                                                block_id=1)
        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=6,
                                                out_channels=24,
                                                repeats=2,
                                                stride=2,
                                                weight_decay=weigth_decay,
                                                block_id=2)
        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=6,
                                                out_channels=32,
                                                repeats=3,
                                                stride=2,
                                                weight_decay=weigth_decay,
                                                block_id=3)

        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=6,
                                                out_channels=64,
                                                repeats=4,
                                                stride=1,
                                                weight_decay=weigth_decay,
                                                block_id=4)

        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=6,
                                                out_channels=96,
                                                repeats=3,
                                                stride=2,
                                                weight_decay=weigth_decay,
                                                block_id=5)

        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=6,
                                                out_channels=160,
                                                repeats=3,
                                                stride=2,
                                                weight_decay=weigth_decay,
                                                block_id=6)

        x = MoblieNetV2.inverted_residual_block(x=x,
                                                expand=6,
                                                out_channels=320,
                                                repeats=1,
                                                stride=1,
                                                weight_decay=weigth_decay,
                                                block_id=7)
        x = MoblieNetV2.conv_block(inputs=x,
                                   output_num=1280,
                                   the_name='conv2',
                                   expand_alpha=1,
                                   kernal=1,
                                   stride=[1, 1])
        x = tflayer.max_pool2d(inputs=x,
                               kernel_size=(7, 7))
        x = tflayer.flatten(inputs=x)

        '''
        the prediction of the network
        return a 2-D tensor with shape (?, output_num)
        '''
        y = tflayer.fully_connected(inputs=x,
                                    num_outputs=output_num,
                                    activation_fn=tf.nn.tanh)
        return y

    @classmethod
    def inverted_residual_block(cls, x, expand, out_channels, repeats,
                                stride, weight_decay, block_id):
        """
        construct a  Inverted Residual Block
        :param x:  4-D tensor
        :param expand: a constant of float which expand the beginning channels in conv_(block_id)_0
        :param out_channels: 4-D tensor
        :param repeats: the times of residual struct
        :param stride: conv kernal stride
        :param weight_decay:
        :param block_id: the name of this block
        :return: 4D tensor
        """
        with tf.name_scope(name="invert_residual_block_{}".format(str(block_id))):
            with tf.name_scope(name="bottleneck_block_{0}_{1}".format(str(block_id),'0')):
                input_channels = x.get_shape().as_list()[-1]
                x = tf.layers.conv2d(inputs=x,
                                     filters=expand * input_channels,
                                     padding='SAME',
                                     strides=1,
                                     kernel_size=1,
                                     name='conv_%d_0' % block_id)
                x = tflayer.batch_norm(inputs=x,
                                       epsilon=1e-5)
                x = tf.nn.relu6(features=x, name='conv_%d_0_act_1' % block_id)
                x = tf.layers.separable_conv2d(inputs=x,
                                               filters=1,
                                               strides=1,
                                               padding='VAlID',
                                               kernel_size=3,
                                               name='conv_DW_%d_0' % block_id)
                x = tflayer.batch_norm(inputs=x,
                                       epsilon=1e-5)
                x = tf.nn.relu6(features=x, name='conv_%d_0_act_2' % block_id)
                x = tf.layers.conv2d(inputs=x,
                                     filters=out_channels,
                                     kernel_size=1,
                                     padding='SAME',
                                     strides=1,
                                     name='conv_bottleneck_%d_0' % block_id)
                x = tflayer.batch_norm(inputs=x,
                                       epsilon=1e-5)
            for i in range(1, repeats):
                with tf.name_scope(name="bottleneck_block_{0}_{1}".format(str(block_id),'0')):
                    x1 = tf.layers.conv2d(inputs=x,
                                          filters=expand * input_channels,
                                          kernel_size=1,
                                          padding='SAME',
                                          strides=1,
                                          name='conv_%d_%d' % (block_id, i))

                    x1 = tflayer.batch_norm(inputs=x1,
                                            epsilon=1e-5)
                    x1 = tf.nn.relu6(features=x1, name='conv_%d_%d_act_1' % (block_id, i))

                    x1 = tf.layers.separable_conv2d(inputs=x1,
                                                    filters=1,
                                                    kernel_size=3,
                                                    padding='VALID',
                                                    strides=1,
                                                    name='conv_DW_%d_%d' % (block_id, i))
                    x1 = tflayer.batch_norm(inputs=x1,
                                            epsilon=1e-5)
                    x1 = tf.nn.relu6(features=x1,
                                     name='conv_dw_%d_%d_act_2' % (block_id, i))

                    x1 = tf.layers.conv2d(inputs=x,
                                          filters=out_channels,
                                          kernel_size=1,
                                          padding='SAME',
                                          strides=1,
                                          name='conv_bottleneck_%d_%d' % (block_id, i))
                    x1 = tflayer.batch_norm(inputs=x1,
                                            epsilon=1e-5)
                    x = tf.add(x=x,
                               y=x1,
                               name='block_%d_%d_output' % (block_id, i))
            return x

    @classmethod
    def conv_block(cls, inputs: tf.Tensor, output_num, expand_alpha=1, the_name='', kernal=3, stride=None):
        """
        convolution Bolck
        :param stride: a list with 4 length and the first and last elements
         must be 1, 2th and 3th element is the stride size
        :param inputs: 4-D tensor
        :param output_num: the number of channels of ouput tensor
        :param expand_alpha: expand ratio of ouput channels
        :param the_name: layer name
        :param kernal: kernal size in filters
        :return: 4-D tensor
        """
        if stride is None:
            stride = [1, 1, 1, 1]
        # input_channels = inputs.get_shape()[-1]       WHAT THE FUCK IT IS!!!!
        # input_channels = int(str(inputs.get_shape()[-1]))
        output_channels = expand_alpha * output_num
        x = tf.layers.conv2d(inputs=inputs,
                             filters=output_channels,
                             kernel_size=kernal,
                             strides=stride,
                             padding='SAME')
        # x = tf.nn.conv2d(input=inputs,
        #                  filter=[kernal, kernal, input_channels, output_channels],
        #                  strides=stride,
        #                  padding='SAME',
        #                  name=the_name)
        x = tflayer.batch_norm(inputs=x,
                               epsilon=1e-5)
        x = tf.nn.relu6(features=x,
                        name='{}_relu'.format(the_name))
        return x
