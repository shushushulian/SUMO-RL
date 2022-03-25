from __future__ import absolute_import
from __future__ import print_function
import os
import datetime
from shutil import copyfile
from training_simulation import Simulation
from generator import TrafficGenerator
from ddqn import DoubleDQN
from memory import Memory
from utils import import_train_configuration, set_sumo, set_train_path
from visualization import Visualization
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = []
    Y = []
    Z = []
    R = []

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])  # gui=False,max_steos=5400
    path = set_train_path(config['models_path_name'])  # models_path_name = models

    Model = DoubleDQN()

    Visualization = Visualization(
        path,
        dpi=96
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],   # max_steps = 5400
        config['n_cars_generated']   # n_cars_generated = 1000
    )
    Memory = Memory(
        config['memory_size_max'],   # memory_size_max = 50000
        config['memory_size_min']   # memory_size_min = 600
    )

    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],   # 0.75
        config['max_steps'],  # 5400
        config['green_duration'],  # 10
        config['yellow_duration'],  # 4
        config['num_states'],  # 80
        config['num_actions'],  # 4
        config['training_epochs']  # 800
    )
    episode = 0
    timestamp_start = datetime.datetime.now()
    Model.reset()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])
        simulation_time, training_time, w, co, co2, reward = Simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        X.append(w)  # wait time
        Y.append(co)
        Z.append(co2)
        R.append(reward)
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    plt.title("reward is car number")
    plt.subplot(3, 1, 1)
    plt.plot(X, label='waiting time')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(Y, label='sum co')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(Z, label='sum co2')
    plt.show()

    plt.title("Reward")
    plt.plot(R, label="Reward")
    plt.legend()
    plt.show()


    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))