from __future__ import absolute_import
from __future__ import print_function
import os
import datetime
from shutil import copyfile
from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path
import matplotlib.pyplot as plt



if __name__ == "__main__":
    X = []
    Y = []
    Z = []
    R = []

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])  # gui=False,max_steos=5400
    path = set_train_path(config['models_path_name'])   # models_path_name = models

    Model = TrainModel(
        config['num_layers'],   # num_layers = 4
        config['width_layers'],  # width_layers = 400
        config['batch_size'],   # batch_size = 100
        config['learning_rate'],  # learning_rate = 0.001
        input_dim=config['num_states'],   # num_states = 80
        output_dim=config['num_actions']  # num_actions = 4
    )

    Memory = Memory(
        config['memory_size_max'],   # memory_size_max = 50000
        config['memory_size_min']   # memory_size_min = 600
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],   # max_steps = 5400
        config['n_cars_generated']   # n_cars_generated = 1000
    )

    Visualization = Visualization(
        path, 
        dpi=96
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
    f = open("/3实验/6/Deep-QLearning-Agent-for-Traffic-Signal-Control-master/wait.txt", 'w')
    g = open("/3实验/6/Deep-QLearning-Agent-for-Traffic-Signal-Control-master/co.txt", 'w')

    f.truncate()
    g.truncate()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time, w, co, co2, reward = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        X.append(w)  # wait time
        Y.append(co)
        Z.append(co2)
        R.append(reward)
        f.write("%s\n" % w)
        g.write("%s\n" % co)
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

    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')