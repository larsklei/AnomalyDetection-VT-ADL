import tensorflow as tf
import keras_core as keras

from typing import Union, Optional

from vit_ae.custom_model.visiontransformer_autoencoder import VisionTransformerAutoencoder
from vit_ae.custom_model.visiontransformer_layers import VisionTransformerBlock, get_decoder

IMG_HEIGHT = 512
IMG_WIDTH = 512
OPTIMIZER_OPTIONS = ["adam", "adamw", "adadelta", "adagrad", "sgd", "rmsprop"]


@keras.saving.register_keras_serializable()
class WarmupScheduler(keras.optimizers.schedules.LearningRateSchedule):
	def __init__(
			self,
			after_warmup_lr_sched: Union[keras.optimizers.schedules.LearningRateSchedule, float],
			warmup_steps: int = 1000,
			warmup_learning_rate: float = 0.00001,
			name: Optional[str] = None
	):
		"""
		
		Args:
			after_warmup_lr_sched:
			warmup_steps:
			warmup_learning_rate:
			name:
		"""
		super().__init__()
		self.name = name
		self.after_warmup_lr_sched = after_warmup_lr_sched
		self.init_warmup_lr = warmup_learning_rate
		self.warmup_steps = warmup_steps
		if isinstance(after_warmup_lr_sched, keras.optimizers.schedules.LearningRateSchedule):
			self.final_warmup_lr = after_warmup_lr_sched(warmup_steps)
		else:
			self.final_warmup_lr = tf.cast(after_warmup_lr_sched, dtype=tf.float32)
	
	def __call__(self, step):
		global_step = tf.cast(step, tf.float32)
		
		linear_warmup_lr = (
				self.init_warmup_lr + global_step / self.warmup_steps * (self.final_warmup_lr - self.init_warmup_lr)
		)
		
		if isinstance(self.after_warmup_lr_sched, keras.optimizers.schedules.LearningRateSchedule):
			after_warmup_lr = self.after_warmup_lr_sched(step)
		else:
			after_warmup_lr = tf.cast(self.after_warmup_lr_sched, tf.float32)
		
		lr = tf.cond(
			global_step < self.warmup_steps,
			lambda: linear_warmup_lr,
			lambda: after_warmup_lr
		)
		
		return lr
	
	def get_config(self):
		if isinstance(self.after_warmup_lr_sched,
		              keras.optimizers.schedules.LearningRateSchedule):
			config = {
				"after_warmup_lr_sched": self.after_warmup_lr_sched,
				"warmup_steps": self.warmup_steps,
				"warmup_learning_rate": self.init_warmup_lr,
				"name": self.name
			}
		else:
			config = {
				"after_warmup_lr_sched": self.after_warmup_lr_sched,
				"warmup_steps": self.warmup_steps,
				"warmup_learning_rate": self.init_warmup_lr,
				"name": self.name
			}
		
		return config
	
	@classmethod
	def from_config(cls, config):
		after_warmup_lr_sched = config["after_warmup_lr_sched"]
		if isinstance(after_warmup_lr_sched,
		              keras.optimizers.schedules.LearningRateSchedule):
			config["after_warmup_lr_sched"] = keras.saving.deserialize_keras_object(config["after_warmup_lr_sched"])
		
		return cls(**config)


def create_optimizer(
		optimizer_name: str,
		learning_rate: Union[keras.optimizers.schedules.LearningRateSchedule, float]
):
	lr_scheduler = WarmupScheduler(
		after_warmup_lr_sched=learning_rate
	)
	if not set([optimizer_name]).issubset(OPTIMIZER_OPTIONS):
		raise ValueError("Wrong optimizer name.")
	if optimizer_name == "adam":
		optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)
	elif optimizer_name == "sgd":
		optimizer = keras.optimizers.SGD(learning_rate=lr_scheduler)
	elif optimizer_name == "adamw":
		optimizer = keras.optimizers.AdamW(learning_rate=lr_scheduler)
	elif optimizer_name == "adadelta":
		optimizer = keras.optimizers.Adadelta(learning_rate=lr_scheduler)
	elif optimizer_name == "adagrad":
		optimizer = keras.optimizers.Adagrad(learning_rate=lr_scheduler)
	elif optimizer_name == "rmsprop":
		optimizer = keras.optimizers.RMSprop(learning_rate=lr_scheduler)
	return optimizer


def create_model(
		embed_dim: int,
		patch_size: int,
		num_heads: int,
		encoder_hidden_units: list[int],
		optimizer: keras.Optimizer,
		num_register: int | None = None,
		kernel_init='glorot_uniform'
):
	encoder = keras.Sequential()
	
	if isinstance(encoder_hidden_units, int):
		encoder_hidden_units = [encoder_hidden_units]
	
	for unit in encoder_hidden_units:
		encoder.add(VisionTransformerBlock(
			num_heads=num_heads,
			embed_dim=embed_dim,
			hidden_unit=unit
		))
	
	decoder = get_decoder(kernel_initializer=kernel_init)
	
	model = VisionTransformerAutoencoder(
		embed_dim=embed_dim,
		patch_size=patch_size,
		encoder=encoder,
		decoder=decoder,
		img_height=IMG_HEIGHT,
		img_width=IMG_WIDTH,
		num_register=num_register
	)
	
	model.compile(
		optimizer=optimizer
	)
	
	return model
