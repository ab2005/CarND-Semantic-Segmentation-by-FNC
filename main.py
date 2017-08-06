import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
import numpy as np
import scipy

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Implementing the function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # print ('Loader file available:', tf.saved_model.loader.maybe_saved_model_directory(vgg_path))
    with sess.as_default():
        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

        # for element in sess.graph.get_operations():
        #    print(element.name)

        image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        # to display in tensorboard
        file_writer = tf.summary.FileWriter('./logs/model')
        file_writer.add_graph(sess.graph)

        # for element in sess.graph.get_operations():
        #    if element.name in ['image_input','layer7_out','layer4_out','layer3_out','keep_prob']:
        #        print (element)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # VGG16
    encoder = vgg_layer7_out

    encoder = tf.layers.conv2d(encoder, num_classes, 1, strides=(1, 1))

    skip4_1x1 = tf.layers.conv2d_transpose(vgg_layer4_out, num_classes, 1, strides=(1, 1), padding='same')

    decoder = tf.add(tf.layers.conv2d_transpose(encoder, num_classes, 4, strides=(2, 2), padding='same'), skip4_1x1)

    decoder = tf.layers.conv2d_transpose(decoder, num_classes, 4, strides=(2, 2), padding='same')

    skip3_1x1 = tf.layers.conv2d_transpose(vgg_layer3_out, num_classes, 1, strides=(1, 1), padding='same')

    decoder = tf.add(decoder, skip3_1x1)

    output = tf.layers.conv2d_transpose(decoder, num_classes, 16, strides=(8, 8), padding='same')

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Setting reporting frequency
    report = 10

    # Limiting the examples in an epoch (experimental)
    limit_up_to_example = 10 ** 6

    # Initialization of previous loss
    previous_epoch_loss = 10**9

    # Setting learning rate and keep probability
    lr = 0.00001
    prob = 0.85

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # Initialization of loss counters
            cumulative_loss = 0
            epoch_loss = 0
            for steps, (images, labels) in enumerate(get_batches_fn(batch_size)):

                _, loss = sess.run([train_op, cross_entropy_loss],
                                   feed_dict={input_image: images, correct_label: labels, learning_rate: lr, keep_prob: prob})

                cumulative_loss += loss
                epoch_loss += loss

                if steps % report == 0 and steps != 0:
                    print("Epoch %i/%i : %0.4f Loss for last %i images" % ((e+1), epochs, cumulative_loss / float(report), len(images)*report))
                    cumulative_loss = 0

                if steps >= limit_up_to_example:
                    break

            print ("Average loss per image in epoch %i has been: %0.4f" % ((e+1), epoch_loss/(len(images)*float(steps))))
            if epoch_loss > previous_epoch_loss:
                print ("Not improving over the previous epoch, early stopping")
                break
            else:
                previous_epoch_loss = epoch_loss

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    USE_GPU = True

    config = tf.ConfigProto(
        device_count={'GPU': 1 if USE_GPU else 0}
    )

    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Building NN using load_vgg, layers, and optimize function

        correct_label = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, os.path.join(data_dir, "vgg"))

        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Training NN using the train_nn function

        # Epochs 20 = Average loss per image in epoch 20 has been: 0.1744

        epochs = 30
        batch_size = 1
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 image_input, correct_label, keep_prob, learning_rate)

        # Saving inference data using helper.save_inference_samples

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Applying the trained model to a video

        def process_videoimage(original_image, sess=sess, image_shape=image_shape, logits=logits, keep_prob=keep_prob, image_input=image_input):
            original_image_shape = original_image.shape

            image = scipy.misc.imresize(original_image, image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_input: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            return np.array(scipy.misc.imresize(street_im, original_image_shape))

        clip1 = VideoFileClip("./data/harder_challenge_video.mp4")
        white_clip = clip1.fl_image(process_videoimage)  # NOTE: this function expects color images!!
        white_clip.write_videofile("./data/segmented_project_video.mp4", audio=False)

if __name__ == '__main__':
    run()
