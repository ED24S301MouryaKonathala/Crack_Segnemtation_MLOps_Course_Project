import yaml

def load_params(path="params.yaml"):
    with open(path) as file:
        params = yaml.safe_load(file)
    return params
