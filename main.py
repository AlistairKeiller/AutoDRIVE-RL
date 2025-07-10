from mlagents_envs.environment import UnityEnvironment  # type: ignore
from gym_unity.envs import UnityToGymWrapper  # type: ignore
from stable_baselines3 import PPO

if __name__ == "__main__":
    unity_env = UnityEnvironment()
    env = UnityToGymWrapper(unity_env, 0, uint8_visual=True)  # type: ignore
    model = PPO("MlpPolicy", env, verbose=1).learn(10_000)
