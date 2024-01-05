import gym
import argparse
from train import MountainCarAgent

def parse_arguments():
    parser = argparse.ArgumentParser(description='MountainCar Agent')
    parser.add_argument('--model_path', type=str, default='your_model_weights.pth', help='Path to the model weights file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    agent = MountainCarAgent()
    agent.load(args.model_path)

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
