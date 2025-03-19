"""Step through an instance of DiffusionModel."""

from diffusion import DiffusionModel, load_config

print("# Initialize model")
params = load_config("config.yaml")
m = DiffusionModel(params)
m.sim_setup()

print("\n# Show initial particle distribution")
print(m.histogram)

print("\n# Step through model")
# while m.running:
for _ in range(m.p.steps):
    m.sim_step()
    print(f"Time = {m.t}")
m.end()

print("\n# Show final particle distribution")
print(m.histogram)
