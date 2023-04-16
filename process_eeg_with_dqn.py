import json
import math
import socket

import torch

from agents.DQN import Model
from utils.hyperparameters import Config

# Config/Hyperparameters
config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epsilon variables
config.epsilon_start    = 0.9
config.epsilon_final    = 0.05
config.epsilon_decay    = 80
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA = 0.9
config.LR    = 0.1

#memory
config.EXP_REPLAY_SIZE = 100000
config.BATCH_SIZE      = 32

#Learning control variables
config.LEARN_START = 40
config.UPDATE_FREQ = 1
config.TARGET_NET_UPDATE_FREQ = 1

# Model - DQN
model = Model(env=None, config=config)

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 8888))
    server.listen(5)
    while True:
        print("Accepting incoming connection!")

        connection, address = server.accept()
        frame_idx = 0
        prev_observation = None
        prev_action = None
        while True:
            data = connection.recv(4096)
            if not data:
                break

            from_client = data.decode("utf-8")
            obj = json.loads(from_client)
            print(obj)

            observation = obj["state"]
            reward = obj["reward"]

            epsilon = config.epsilon_by_frame(frame_idx)
            action = model.get_action(observation, epsilon)
            print("DQN result: {0}".format(action))

            if prev_observation is None:
                prev_observation = observation
                prev_action = action

            model.update(prev_observation, prev_action, reward, observation, frame_idx)

            prev_observation = observation
            prev_action = action

            frame_idx += 1

            connection.send(str(action).encode())
        print("Closing incoming connection!")
        connection.close()

if __name__ == "__main__":
    main()
