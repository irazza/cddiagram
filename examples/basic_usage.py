from cddiagram import draw_cd_diagram
import numpy as np


rng = np.random.default_rng(1)

models = {
    'model1': rng.normal(loc=0.2, scale=0.1, size=30),
    'model2': rng.normal(loc=0.2, scale=0.1, size=30),
    'model3': rng.normal(loc=0.4, scale=0.1, size=30),
    'model4': rng.normal(loc=0.5, scale=0.1, size=30),
    'model5': rng.normal(loc=0.7, scale=0.1, size=30),
    'model6': rng.normal(loc=0.7, scale=0.1, size=30),
    'model7': rng.normal(loc=0.8, scale=0.1, size=30),
    'model8': rng.normal(loc=0.9, scale=0.1, size=30),
}
samples = np.column_stack([models[k] for k in models])
draw_cd_diagram(samples, labels=list(models.keys()), out_file="out.svg", title="Model comparison")