import gym
from train import MountainCarAgent

if __name__ == "__main__":
    agent = MountainCarAgent()

    agent.load('your_model_weights.pth')

    env = gym.make("MountainCar-v0")
    state = env.reset()

    try:
        while True:
            env.render()
            action = agent.select_action(state, 0, agent.get_model())
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
    except KeyboardInterrupt:
        print("Environment closed.")
        env.close()
