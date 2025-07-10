from mlagents_envs.environment import UnityEnvironment  # type: ignore
from gym_unity.envs import UnityToGymWrapper  # type: ignore
from stable_baselines3 import PPO

if __name__ == "__main__":
    print("Starting PPO training script...")
    unity_env = UnityEnvironment()
    env = UnityToGymWrapper(unity_env)  # type: ignore
    model = PPO("MlpPolicy", env, verbose=1).learn(10_000)
