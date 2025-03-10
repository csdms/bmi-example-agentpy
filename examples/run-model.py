"""Run an instance of DiffusionModel."""

from diffusion import DiffusionModel, load_config

print("# Run model")

params = load_config("config.yaml")
m = DiffusionModel(params)
results = m.run(steps=1)

print("\n# Model results:")
print(results)

print("\n# Final particle distribution:")
print(m.histogram)
