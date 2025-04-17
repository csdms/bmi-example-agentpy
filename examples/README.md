# examples

Examples of running the AgentPy `DiffusionModel`
standalone and through a Basic Model Interface.

## Configuration files

* [config.yaml](./config.yaml): Stores a set of parameters for `DiffusionModel` and its BMI.
* [config-big.yaml](./config-big.yaml): Another configuration file, but with more agents on a larger grid.

## Scripts

* [step-model.py](./step-model.py): Advances an instance of `DiffusionModel` one time step at a time.
* [run-model.py](./run-model.py): Runs an instance of `DiffusionModel`.
* [step-bmi-model.py](./step-bmi-model.py): Exercises the `DiffusionModel` BMI, printing information to stdout.
* [run-bmi-model.py](./run-bmi-model.py): Runs an instance of `DiffusionModel` through its BMI.

## Notebooks

* [explore-model.ipynb](./explore-model.ipynb): Explores the parameters and initial conditions of a `DiffusionModel` instance.
* [run-model.ipynb](./run-model.ipynb): Shows how to run `DiffusionModel` standalone and make a plot of the results.
