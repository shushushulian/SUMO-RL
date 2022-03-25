import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        生成每次仿真的车辆
        """
        np.random.seed(seed)  # 使得每次产生的车辆情况相同

        # 车辆生成服从weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)  # 给timings排序

        # 重新调整分布以适应间隔 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) +  max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
            <vType  accel="1.0" decel="4.5" id="BUS" minGap="2.5" maxSpeed="25" sigma="0.5" vClass="bus" color="0,0,1" />
            <vType  accel="1.0" decel="4.5" id="truck" minGap="2.5" maxSpeed="25" sigma="0.5" vClass="truck" color="0,1,0" />
            <vType  accel="1.0" decel="4.5" id="evehicle" minGap="2.5" maxSpeed="25" sigma="0.5" vClass="evehicle" color="1,0,0" />
            <vType  accel="1.0" decel="4.5" id="motorcycle" minGap="2.5" maxSpeed="25" sigma="0.5" vClass="motorcycle" color="1,0,1" />


            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                car_class = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="W_E%i" type="evehicle" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="W_E_%i" type="truck" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="W_E_%i" type="BUS" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="W_E%i" type="motorcycle" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                    elif route_straight == 2:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="E_W%i" type="evehicle" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="E_W_%i" type="truck" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="E_W_%i" type="BUS" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="E_W%i" type="motorcycle" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_straight == 3:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="N_S%i" type="evehicle" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="N_S_%i" type="truck" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="N_S_%i" type="BUS" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="N_S%i" type="motorcycle" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    else:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:

                            print(
                                '    <vehicle id="S_N%i" type="evehicle" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="S_N_%i" type="truck" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="S_N_%i" type="BUS" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="S_N%i" type="motorcycle" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="W_N%i" type="evehicle" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="W_N_%i" type="truck" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="W_N_%i" type="BUS" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="W_N%i" type="motorcycle" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 2:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="W_S%i" type="evehicle" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="W_S_%i" type="truck" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="W_S_%i" type="BUS" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="W_S%i" type="motorcycle" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 3:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="N_W%i" type="evehicle" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="N_W_%i" type="truck" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="N_W_%i" type="BUS" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="N_W%i" type="motorcycle" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 4:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="N_E%i" type="evehicle" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="N_E_%i" type="truck" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="N_E_%i" type="BUS" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="N_E%i" type="motorcycle" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 5:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="E_N%i" type="evehicle" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="E_N_%i" type="truck" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="E_N_%i" type="BUS" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="E_N%i" type="motorcycle" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 6:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="E_S%i" type="evehicle" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="E_S_%i" type="truck" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="E_S_%i" type="BUS" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="E_S%i" type="motorcycle" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 7:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="S_W%i" type="evehicle" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="S_W_%i" type="truck" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="S_W_%i" type="BUS" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="S_W%i" type="evehicle" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

                    elif route_turn == 8:
                        if 0 < car_class < 0.3:
                            print(
                                '    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.3 < car_class < 0.6:
                            print(
                                '    <vehicle id="S_E%i" type="evehicle" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.6 < car_class < 0.75:
                            print(
                                '    <vehicle id="S_E_%i" type="BUS" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        elif 0.75 < car_class < 0.9:
                            print(
                                '    <vehicle id="S_E_%i" type="BUS" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)
                        else:
                            print(
                                '    <vehicle id="S_E%i" type="evehicle" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                    car_counter, step), file=routes)

            print("</routes>", file=routes)