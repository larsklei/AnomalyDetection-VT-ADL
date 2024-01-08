import logging
import os
import tensorflow as tf
import click
import mlflow
import keras_core as keras

from vit_ae.train.train_utilities import create_optimizer, create_model
from vit_ae.preprocess.preprocess_utilities import get_image_files, get_train_val_split, DataGenerator

AUTOTUNE = tf.data.AUTOTUNE
OPTIMIZER_OPTIONS = ["adam", "adamw", "adadelta", "adagrad", "sgd", "rmsprop"]
OBJECT_NAMES = [
	"all",
	"bottle",
	"cable",
	"capsule",
	"carpet",
	"grid",
	"hazelnut",
	"leather",
	"metal_nut",
	"pill",
	"screw",
	"tile",
	"toothbrush",
	"transistor",
	"wood",
	"zipper"
]


@click.command()
@click.option("--source_path", required=True, type=click.Path(exists=True, file_okay=False),
              help="Path to the MVTEC AD dataset")
@click.option("--data_objects", default="bottle", type=click.Choice(OBJECT_NAMES), help="Objects which are used.")
@click.option("--val_split", default=None, type=click.FLOAT, help="Validation split.")
@click.option("--epochs", default=1, type=click.INT, help="Number of epochs.")
@click.option("--embed_dim", default=512, type=click.INT, help="Embedding Dimension.")
@click.option("--num_heads", default=4, type=click.INT, help="Number of heads used in Attention of the ViTBlocks.")
@click.option("--hidden_unit", "-unit", default=(32, 32, 32), multiple=True, type=click.INT, help="Number of units used in ViTBlocks.")
@click.option("--img_height", default=512, type=click.INT)
@click.option("--img_width", default=512, type=click.INT)
@click.option("--patch_size", default=16, type=click.INT, help="Size of the patches.")
@click.option("--num_register", default=None, type=click.INT, help="Number of registers")
@click.option("--batch_size", default=32, type=click.INT)
@click.option("--optimizer", default="adam",
              type=click.Choice(OPTIMIZER_OPTIONS))
@click.option("--lr_rate", default=0.001, type=float)
def perform_train_run(source_path, data_objects, val_split, epochs, embed_dim, num_heads, hidden_unit, img_height,
                      img_width, patch_size, num_register, batch_size, optimizer, lr_rate, ):
	AUTOTUNE = tf.data.AUTOTUNE
	hidden_units = list(hidden_unit)
	train_image_files = get_image_files(source_path, data_objects, training=True)
	output_signature = (
		tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
		tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32)
	)
	
	optimizer = create_optimizer(optimizer_name=optimizer, learning_rate=lr_rate)
	model = create_model(
		embed_dim=embed_dim,
		patch_size=patch_size,
		num_heads=num_heads,
		encoder_hidden_units=hidden_units,
		optimizer=optimizer,
		num_register=num_register,
		kernel_init='glorot_uniform'
	)
	if val_split is not None:
		train_image_files, val_image_files = get_train_val_split(
			train_image_files, validation_split=val_split
		)
		steps_per_epoch = len(train_image_files) // batch_size + 1
		val_steps = len(val_image_files) // batch_size + 1
		train_ds = tf.data.Dataset.from_generator(
			DataGenerator(train_image_files, training=True),
			output_signature=output_signature
		)
		val_ds = tf.data.Dataset.from_generator(
			DataGenerator(val_image_files, training=True),
			output_signature=output_signature
		)
		train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) \
			.repeat().batch(batch_size, drop_remainder=True)
		val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) \
			.repeat().batch(batch_size, drop_remainder=True)
		model.fit(
			x=train_ds,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=val_ds,
			validation_steps=val_steps
		)
	else:
		steps_per_epoch = len(train_image_files) // batch_size + 1
		train_ds = tf.data.Dataset.from_generator(
			DataGenerator(train_image_files, training=True),
			output_signature=output_signature
		)
		train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) \
			.repeat().batch(batch_size, drop_remainder=True)
		model.fit(
			x=train_ds,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch
		)
	logging.info("Model is fitted on training set.")
	test_image_files = get_image_files(source_path, data_objects, training=False)
	training_steps = int(len(test_image_files) // batch_size + 1)
	test_ds = tf.data.Dataset.from_generator(
		DataGenerator(image_files=test_image_files, training=False),
		output_signature=output_signature
	)
	test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat().batch(batch_size, drop_remainder=True)
	score = model.evaluate(test_ds, steps=training_steps)
	
	print(score)


if __name__ == "__main__":
	perform_train_run()
