MODELS = {}

def register_model(name):
    def register_curr_model(model_class):
        MODELS[name] = model_class
        return model_class
    return register_curr_model


def load_model(model_config):
    return MODELS[model_config.name](**model_config)