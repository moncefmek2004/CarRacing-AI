
import gymnasium as gym

env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    max_episode_steps=3000,
)

try:
    observation, info = env.reset(seed=202)

    total_reward = 0.0

    # Execute at most 100 random actions.
    for step in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        if (step + 1) % 10 == 0:
            print(
                f"Step: {step + 1}, "
                f"Reward: {reward:.2f}, "
                f"Total: {total_reward:.2f}"
            )

        if terminated or truncated:
            print("Episode ended.")
            break

    print("Final total reward:", round(total_reward, 2))

finally:
    env.close()
