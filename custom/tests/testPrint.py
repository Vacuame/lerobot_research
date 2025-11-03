from lerobot.policies.customACT.configuration_customACT import ACTConfig

# Instantiate ACTConfig and print a few properties for verification
cfg = ACTConfig()
print("Created ACTConfig:", cfg)
print("n_history_obs_states:", cfg.n_history_obs_states)
print("history_action_delta_indices length:", len(cfg.history_action_delta_indices))
print("first 5 history_action_delta_indices:", cfg.history_action_delta_indices[:5])
