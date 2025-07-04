from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class InspectRGBPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 500
    experiment_name = "InspectRobot"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[1024,512,256],
        critic_hidden_dims=[1024,512, 256],

        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

