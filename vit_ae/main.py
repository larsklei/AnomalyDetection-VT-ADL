import tensorflow as tf

from train.train_utilities import create_optimizer, create_model
from preprocess.preprocess_utilities import get_image_files, DataGenerator

batch_size = 32
data_objects = "bottle"
img_height = 512
img_width = 512
optimizer = "adam"
lr_rate=0.001
epochs=1
embed_dim = 32
patch_size = 16
num_heads = 2
encoder_hidden_units = [32, 16]
num_register = 1

source_path = "/Users/larskleinemeier/Documents/AnomalyDetection-VT-ADL/vit_ae/data_images"
image_files = get_image_files(source_path, data_objects, training=True)
output_signature = (
	tf.TensorSpec(shape = (img_height, img_width, 3), dtype=tf.float32),
	tf.TensorSpec(shape = (img_height, img_width, 3), dtype=tf.float32)
)

optimizer = create_optimizer(optimizer_name=optimizer, learning_rate=lr_rate)
model = create_model(
	embed_dim=embed_dim,
	patch_size=patch_size,
	num_heads=num_heads,
	encoder_hidden_units=encoder_hidden_units,
	optimizer=optimizer,
	num_register=num_register,
	kernel_init = 'glorot_uniform'
)


steps_per_epoch = len(image_files)//batch_size+1
AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_generator(
	DataGenerator(image_files, training=True),
	output_signature=output_signature
)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat()
train_ds = train_ds.batch(32, drop_remainder=True)
model.fit(
	x=train_ds,
	epochs=epochs,
	steps_per_epoch=steps_per_epoch
)

test_image_files = get_image_files(source_path, data_objects, training=False)
training_steps= int(len(test_image_files)//batch_size+1)
test_ds = tf.data.Dataset.from_generator(
	DataGenerator(image_files=test_image_files, training=False),
	output_signature=output_signature
)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat().batch(batch_size, drop_remainder=True)
score = model.evaluate(test_ds, return_dict=True, steps=training_steps)

config = model.get_config()
print(config)




