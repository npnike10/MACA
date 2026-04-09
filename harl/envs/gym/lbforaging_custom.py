try:
    from gym.envs.registration import register as gym_register
    from gym.envs.registration import registry as gym_registry
except ImportError:
    gym_register = None
    gym_registry = None

try:
    from gymnasium.envs.registration import register as gymnasium_register
    from gymnasium.envs.registration import registry as gymnasium_registry
except ImportError:
    gymnasium_register = None
    gymnasium_registry = None


CUSTOM_FORAGING_SPECS = [
    (
        "Foraging-15x15-3p-5f-coop-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (15, 15),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 15,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-15x15-3p-5f-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (15, 15),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 15,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-15x15-3p-5f-2s-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (15, 15),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-15x15-4p-5f-v3",
        {
            "players": 4,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (15, 15),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 15,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-15x15-4p-5f-coop-v3",
        {
            "players": 4,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (15, 15),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 15,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-20x20-5p-5f-v3",
        {
            "players": 5,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (20, 20),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 20,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-20x20-3p-5f-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (20, 20),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 20,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-2s-9x9-3p-2f-coop-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (9, 9),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 2,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-2s-9x9-3p-2f-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (9, 9),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 2,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-2s-11x11-3p-2f-coop-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (11, 11),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 2,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-2s-11x11-3p-2f-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (11, 11),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 2,
            "sight": 2,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-4s-11x11-3p-2f-coop-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (11, 11),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 2,
            "sight": 4,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-4s-11x11-3p-2f-v3",
        {
            "players": 3,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (11, 11),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 2,
            "sight": 4,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-7s-20x20-5p-3f-coop-v3",
        {
            "players": 5,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (20, 20),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 3,
            "sight": 7,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-7s-20x20-5p-3f-v3",
        {
            "players": 5,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (20, 20),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 3,
            "sight": 7,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-8s-25x25-8p-5f-v3",
        {
            "players": 8,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (25, 25),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 5,
            "sight": 8,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
    (
        "Foraging-7s-30x30-7p-4f-v3",
        {
            "players": 7,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (30, 30),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": 4,
            "sight": 7,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    ),
]


def _is_registered(env_id, registry):
    if registry is None:
        return False
    if hasattr(registry, "env_specs"):
        return env_id in registry.env_specs
    return env_id in registry


def register_custom_lbforaging_envs():
    import lbforaging  # noqa: F401

    for env_id, env_kwargs in CUSTOM_FORAGING_SPECS:
        if gym_register is not None and not _is_registered(env_id, gym_registry):
            gym_register(
                id=env_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs=env_kwargs,
            )
        if (
            gymnasium_register is not None
            and not _is_registered(env_id, gymnasium_registry)
        ):
            gymnasium_register(
                id=env_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs=env_kwargs,
            )
