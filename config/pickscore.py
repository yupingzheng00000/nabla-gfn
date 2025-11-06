from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "pap"  # Pick-a-Pic prompts (matches PickScore training data)
    config.experiment.reward_fn = "pickscore"
    config.experiment.prompt_fn_kwargs = {}
    
    # PickScore has smaller output range, requires higher reward_scale
    # Teacher's baseline: pickscore: 5e5 (vs aesthetic: 1e4, hps: 3e6)
    config.model.reward_scale = 5e5
    
    return config
