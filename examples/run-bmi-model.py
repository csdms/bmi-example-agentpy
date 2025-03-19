"""Run DiffusionModel with only BMI calls."""
import numpy as np

from diffusion import BmiDiffusionModel

CONFIG_FILE = "config.yaml"


print("Create an instance of BmiDiffusionModel.")
m = BmiDiffusionModel()
print(m.get_component_name())

print("Initialize the model.")
m.initialize(CONFIG_FILE)
print(CONFIG_FILE)

var_name = m.get_output_var_names()[0]
grid_id = m.get_var_grid(var_name)
grid_size = m.get_grid_size(grid_id)

print("Get the initial values of the particle histogram.")
val = np.empty(grid_size, dtype=m.get_var_type(var_name))
m.get_value(var_name, val)
print(f" - values at time {m.get_current_time()}:")
print(val)

print("Run the model to completion.")
while m.get_current_time() < m.get_end_time():
    m.update()
print(" - new time:", m.get_current_time())

print("Get the final values of the particle histogram.")
val = np.empty(grid_size, dtype=m.get_var_type(var_name))
m.get_value(var_name, val)
print(f" - values at time {m.get_current_time()}:")
print(val)

print("Finalize the model.")
m.finalize()
print("Done.")
