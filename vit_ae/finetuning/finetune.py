import optuna


def create_model(trial):
	pass

def create_optimizer(trial):
	pass

def objective(trial):
	pass

if __name__ == "__main__":
	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=100)

