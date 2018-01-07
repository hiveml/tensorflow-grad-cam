import os,sys
import tensorflow as tf
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import cv2

from model.nets import nets_factory
from model.preprocessing import preprocessing_factory

flags = tf.app.flags
flags.DEFINE_string("input", "images/cat.jpg", "Path to input image ['images/cat.jpg']")
flags.DEFINE_string("output", "output.png", "Path to output image ['output.png']")
flags.DEFINE_string("layer_name", None, "Layer till which to backpropagate")
flags.DEFINE_string("model_name", "resnet_v2_50", "Name of the model")
flags.DEFINE_string("preprocessing_name", None, "Name of the image preprocessor")
flags.DEFINE_integer("eval_image_size", None, "Resize images to this size before eval")
flags.DEFINE_string("dataset_dir", "./imagenet", "Location of the labels.txt")
flags.DEFINE_string("checkpoint_path", "./imagenet/resnet_v2_50.ckpt", "saved weights for model")
flags.DEFINE_integer("label_offset", 1, "Used for imagenet with 1001 classes for background class")

FLAGS = flags.FLAGS

slim = tf.contrib.slim

_layer_names = { "resnet_v2_50":       ["PrePool","predictions"],
                 "resnet_v2_101":       ["PrePool","predictions"],
                 "resnet_v2_152":       ["PrePool","predictions"],
                 }

_logits_name = "Logits"

def load_labels_from_file(dataset_dir):
  labels = {}
  labels_name = os.path.join(dataset_dir,'labels.txt')
  with open(labels_name) as label_file:
    for line in label_file:
      idx,label = line.rstrip('\n').split(':')
      labels[int(idx)] = label
  assert len(labels) > 1 
  return labels


def load_image(img_path):
  print("Loading image")
  img = cv2.imread(img_path)
  if img is None:
      sys.stderr.write('Unable to load img: %s\n' % img_path)
      sys.exit(1)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  return img


def preprocess_image(image,eval_image_size):
  preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          preprocessing_name, is_training=False)
  image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
  return image

def grad_cam(img, imgs0, end_points, sess, predicted_class, layer_name, nb_classes, eval_image_size):
  # Conv layer tensor [?,10,10,2048]
  conv_layer = end_points[layer_name]
  # [1000]-D tensor with target class index set to 1 and rest as 0
  one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
  signal = tf.multiply(end_points[_logits_name], one_hot)
  loss = tf.reduce_mean(signal)

  grads = tf.gradients(loss, conv_layer)[0]
  # Normalizing the gradients
  norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

  output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={imgs0: img})
  output = output[0]           # [10,10,2048]
  grads_val = grads_val[0]	 # [10,10,2048]

  weights = np.mean(grads_val, axis = (0, 1)) 			# [2048]
  cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [10,10]

  # Taking a weighted average
  for i, w in enumerate(weights):
    cam += w * output[:, :, i]

  # Passing through ReLU
  cam = np.maximum(cam, 0)
  cam = cam / np.max(cam)
  cam3 = cv2.resize(cam, (eval_image_size,eval_image_size))

  return cam3


def main(_):
  checkpoint_path=FLAGS.checkpoint_path
  img = load_image(FLAGS.input)

  labels = load_labels_from_file(FLAGS.dataset_dir)
  num_classes = len(labels) + FLAGS.label_offset

  network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=num_classes,
    is_training=False)

  eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

  print("\nLoading Model")
  imgs0 = tf.placeholder(tf.uint8, [None,None, 3])
  imgs = preprocess_image(imgs0,eval_image_size)
  imgs = tf.expand_dims(imgs,0)

  _,end_points = network_fn(imgs)

  init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())

  print("\nFeedforwarding")

  with tf.Session() as sess:
    init_fn(sess)

    ep = sess.run(end_points, feed_dict={imgs0: img})
    pred_layer_name = _layer_names[FLAGS.model_name][1]
    probs = ep[pred_layer_name][0]

    preds = (np.argsort(probs)[::-1])[0:5]
    print('\nTop 5 classes are')
    for p in preds:
        print(labels[p-FLAGS.label_offset], probs[p])

    # Target class
    predicted_class = preds[0]
    # Target layer for visualization
    layer_name = FLAGS.layer_name or _layer_names[FLAGS.model_name][0]
    # Number of output classes of model being used
    nb_classes = num_classes

    cam3 = grad_cam(img, imgs0, end_points, sess, predicted_class, layer_name, nb_classes, eval_image_size)

    img = cv2.resize(img,(eval_image_size,eval_image_size))
    img = img.astype(float)
    img /= img.max()


    cam3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)

    # Superimposing the visualization with the image.
    alpha = 0.0025
    new_img = img+alpha*cam3
    new_img /= new_img.max()

    # Display and save
    io.imshow(new_img)
    plt.axis('off')
    plt.savefig(FLAGS.output,bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
	tf.app.run()

