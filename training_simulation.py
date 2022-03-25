# state为排放等级矩阵
import traci
import numpy as np
import random
import timeit
import os

PHASE_NS_GREEN = 0  # action 0 code 00，南北绿
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01，南北左转
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10，东西
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11，东西左转
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._total_co = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        start_time = timeit.default_timer()

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._wait_car_number = 0
        self._waiting_times = {}
        self._co = {}
        self._hc = {}
        self._nox = {}
        self._co2 = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._wait_time = 0
        self._sum_co = 0
        self._sum_co2 = 0
        self.done = False
        old_total_wait = 0
        old_total_emssion = 0
        # old_total_co = 0
        # old_total_hc = 0
        # old_total_nox = 0
        # old_total_co2 = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # 获取当前路口的state
            current_state = self._get_state()

            # 获取当前状态下交叉口车辆的等待时间，排放数据
            # current_total_wait = self._collect_waiting_car()
            current_total_wait, current_co2, current_hc, curret_nox = self._collect_waiting_times()

            reward_wait = old_total_wait - current_total_wait
            current_total_emssion = (0.046 * current_co2) + (0.15 * current_hc) + (0.804 * curret_nox)
            reward_emssion = old_total_emssion - current_total_emssion
            # 设置奖励
            reward = (0.6 * reward_wait) + (0.4 * reward_emssion)
            # self._Memory.add_sample((old_state, old_action, reward, current_state))

            # 根据交叉口的当前状态选择要激活的灯光相位
            action = self._Model.pick_action(current_state)

            # 如果选择的相位与上一个相位不一样则激活黄色
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # 执行之前选择的阶段
            self._set_green_phase(action)
            self._simulate(self._green_duration)  # green duration = 10, yellow = 4
            if old_state == -1:
                old_state = current_state

            # 将数据保存到memory并且更新网络
            if self._step != 0:
                self._Model.update(old_state, old_action, reward, current_state, self.done, epsilon)

            # 更新变量
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            old_total_emssion = current_total_emssion

            # 只保存有意义的奖励以便更好地查看代理的行为是否正确
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        print("wait time: ", self._sum_waiting_time)
        print("sum co2: ", self._sum_co)
        w = self._sum_waiting_time
        co = self._sum_co
        co2 = self._sum_co2
        reward = self._sum_neg_reward
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time, w, co, co2, reward


    def _simulate(self, steps_todo):
        """
        在收集统计数据时执行sumo中的步骤
        """
        if (self._step + steps_todo) >= self._max_steps:
            self.done = True
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1  # 更新步数
            steps_todo -= 1
            queue_length, sum_co, sum_co2 = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # 对每一辆车来说排队等候的每一秒意味着每辆车等候一秒，因此排队长度=等待秒数
            self._sum_co += sum_co

    def _collect_waiting_car(self):
        """检索在各个车道上等待的车的数量"""
        emssion_class = ['Zero/default', "HBEFA3/LDV_G_EU6", 'HBEFA3/PC_G_EU4', 'HBEFA3/Bus', 'HBEFA3/HDV']
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        w_car = 0
        c_number = 0
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                ve = traci.vehicle.getSpeed(car_id)
                v_class = traci.vehicle.getEmissionClass(car_id)
                if ve <= 0.1:
                    if v_class == emssion_class[0]:
                        c_number = 1
                    elif v_class == emssion_class[1]:
                        c_number = 2
                    elif v_class == emssion_class[2]:
                        c_number = 3
                    elif v_class == emssion_class[3]:
                        c_number = 4
                    elif v_class == emssion_class[4]:
                        c_number = 5
                    else:
                        c_number = 0
                w_car += c_number
            self._wait_car_number = w_car

        return self._wait_car_number



    def _collect_waiting_times(self):
        """
       检索每辆车在进站道路上的等待时间
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        car_numbers = 0
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            co = traci.vehicle.getCOEmission(car_id)
            hc = traci.vehicle.getHCEmission(car_id)
            nox = traci.vehicle.getNOxEmission(car_id)
            co2 = traci.vehicle.getCO2Emission(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # 获得车辆所在的道路id
            if road_id in incoming_roads:  #   consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
                self._co[car_id] = co
                self._hc[car_id] = hc
                self._nox[car_id] = nox
                self._co2[car_id] = co2
                car_numbers += 1
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
                    del self._co[car_id]
                    del self._hc[car_id]
                    del self._nox[car_id]
                    del self._co2[car_id]
        if car_numbers == 0:
            car_numbers = 1

        total_waiting_time = (sum(self._waiting_times.values())) / car_numbers
        co2 = sum(self._co2.values())
        hc = sum(self._hc.values())
        nox = sum(self._nox.values())

        total_co2 = co2 / car_numbers
        total_hc = hc / car_numbers
        total_nox = nox / car_numbers

        return total_waiting_time, total_co2, total_hc, total_nox,
    # , total_co, total_hc, total_nox, total_co2


    def _choose_action(self, state, epsilon):
        """
        根据epsilon贪婪策略，决定是否进行探索性或贪婪策略行动
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)  # 随机行动
        else:
            return np.argmax(self._Model.predict_one(state))  # 当前状态下的最佳行动


    def _set_yellow_phase(self, old_action):
        """
        激活正确的黄灯组合 in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # 根据旧动作获取黄色相位码 (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)   # 南北绿灯
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)  # 南北左转绿灯
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)   # 东西绿灯
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)  # 东西左转绿灯


    def _get_queue_length(self):
        """
        检索每个进入车道中速度为0的车辆数
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")  # 返回给定边上最后一个时间步的停止车辆总数。低于0.1 m / s的速度被认为是停止。
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        co_N = traci.edge.getCOEmission("N2TL")
        co_S = traci.edge.getCOEmission("S2TL")
        co_E = traci.edge.getCOEmission("E2TL")
        co_W = traci.edge.getCOEmission("W2TL")
        co2_N = traci.edge.getCO2Emission("N2TL")
        co2_S = traci.edge.getCO2Emission("S2TL")
        co2_E = traci.edge.getCO2Emission("E2TL")
        co2_W = traci.edge.getCO2Emission("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        sum_co = co_S + co_N + co_W + co_E
        sumo_co2 = co2_S + co2_E + co2_W + co2_N

        return queue_length, sum_co, sumo_co2

    def _get_state(self):
        positionMatrix = []  # 位置矩阵
        velocityMatrix = []  # 速度矩阵
        emssionMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        junctionPosition = traci.junction.getPosition('TL')[0]  # 交叉口位置
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('E2TL')  # 返回上一个模拟步骤中指定边上的车辆ID列表
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('W2TL')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('N2TL')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('S2TL')
        for i in range(16):
            positionMatrix.append([])
            velocityMatrix.append([])
            emssionMatrix.append([])
            for j in range(16):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)
                emssionMatrix[i].append(0)

        for v in vehicles_road1:
            # 计算车辆距离交叉路的距离
            ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
            # print("E car position:", traci.vehicle.getLanePosition(v))
            e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
                v) + traci.vehicle.getNOxEmission(v)
            if (ind < 16):
                positionMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
                emssionMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = e

        for v in vehicles_road2:
            ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
            e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
                v) + traci.vehicle.getNOxEmission(v)
            if (ind < 16):
                positionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
                emssionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = e

        junctionPosition = traci.junction.getPosition('TL')[1]
        for v in vehicles_road3:
            ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
            e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
                v) + traci.vehicle.getNOxEmission(v)
            if (ind < 16):
                positionMatrix[8 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[8 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
                emssionMatrix[8 + traci.vehicle.getLaneIndex(v)][ind] = e

        for v in vehicles_road4:
            ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
            e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
                v) + traci.vehicle.getNOxEmission(v)
            if (ind < 16):
                positionMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
                emssionMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = e

        position = np.array(positionMatrix)
        position = position.reshape(1, 16, 16, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 16, 16, 1)

        emssion = np.array(emssionMatrix)
        emssion = emssion.reshape(1, 16, 16, 1)

        return [position, velocity, emssion]
    # def _get_state(self):
    #     positionMatrix = []  # 位置矩阵
    #     velocityMatrix = []  # 速度矩阵
    #     emssionMatrix = []
    #
    #     cellLength = 7
    #     offset = 11
    #     speedLimit = 14
    #
    #     junctionPosition = traci.junction.getPosition('TL')[0]  # 交叉口位置
    #     vehicles_road1 = traci.edge.getLastStepVehicleIDs('E2TL')  # 返回上一个模拟步骤中指定边上的车辆ID列表
    #     vehicles_road2 = traci.edge.getLastStepVehicleIDs('W2TL')
    #     vehicles_road3 = traci.edge.getLastStepVehicleIDs('N2TL')
    #     vehicles_road4 = traci.edge.getLastStepVehicleIDs('S2TL')
    #     for i in range(16):
    #         positionMatrix.append([])
    #         velocityMatrix.append([])
    #         emssionMatrix.append([])
    #         for j in range(12):
    #             positionMatrix[i].append(0)
    #             velocityMatrix[i].append(0)
    #             emssionMatrix[i].append(0)
    #
    #     for v in vehicles_road1:
    #         ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
    #         # e = traci.vehicle.getEmissionClass(v)
    #         e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
    #             v) + traci.vehicle.getNOxEmission(v)
    #         if (ind < 12):
    #             positionMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = 1
    #             velocityMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
    #             emssionMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = e
    #
    #     for v in vehicles_road2:
    #
    #         ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
    #         e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
    #             v) + traci.vehicle.getNOxEmission(v)
    #         if (ind < 12):
    #             positionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = 1
    #             velocityMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
    #             emssionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = e
    #
    #     junctionPosition = traci.junction.getPosition('TL')[1]
    #     for v in vehicles_road3:
    #         ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
    #         e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
    #             v) + traci.vehicle.getNOxEmission(v)
    #         if (ind < 12):
    #             positionMatrix[8 + traci.vehicle.getLaneIndex(v)][ind] = 1
    #             velocityMatrix[8 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
    #             emssionMatrix[8 + traci.vehicle.getLaneIndex(v)][ind] = e
    #
    #     for v in vehicles_road4:
    #         ind = int(abs(750 - traci.vehicle.getLanePosition(v)) / cellLength)
    #         e = traci.vehicle.getCOEmission(v) + traci.vehicle.getCO2Emission(v) + traci.vehicle.getHCEmission(
    #             v) + traci.vehicle.getNOxEmission(v)
    #         if (ind < 12):
    #             positionMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = 1
    #             velocityMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
    #             emssionMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = e
    #
    #     position = np.array(positionMatrix)
    #     position = position.reshape(1, 16, 12, 1)
    #
    #     velocity = np.array(velocityMatrix)
    #     velocity = velocity.reshape(1, 16, 12, 1)
    #
    #     emssion = np.array(emssionMatrix)
    #     emssion = emssion.reshape(1, 16, 12, 1)
    #
    #     return [position, emssion, velocity]



    def _replay(self):
        batch = self._Memory.get_samples(self._Model.batch_size)  # batch size = 100
        em1 = np.zeros((100, 16, 12, 1))
        v1 = np.zeros((100, 16, 12, 1))
        p1 = np.zeros((100, 16, 12, 1))
        em2 = np.zeros((100, 16, 12, 1))
        v2 = np.zeros((100, 16, 12, 1))
        p2 = np.zeros((100, 16, 12, 1))
        if len(batch) > 0:  # if the memory is full enough

            # states = [val[0] for val in batch]
            # next_states = np.array([val[3] for val in batch])  # extract next states from the batch
            i = 0
            for val in batch:
                p1[i] = val[0][0]
                em1[i] = val[0][1]
                v1[i] = val[0][2]
                p2[i] = val[3][0]
                em2[i] = val[3][1]
                v2[i] = val[3][2]
                i += 1
            states = [p1, em1, v1]
            next_states = [p2, em2, v2]
            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            y = np.zeros((100, 1, 4))
            em = np.zeros((100, 16, 12, 1))
            v = np.zeros((100, 16, 12, 1))
            p = np.zeros((100, 16, 12, 1))
            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                # current_q = np.reshape(current_q, [1, 4])
                p[i] = state[0]
                em[i] = state[1]
                v[i] = state[2]
                y[i] = current_q
                # x.append(np.array(state))
                # y.append(np.array(current_q))
            x = [p, em, v]
            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
       保存事件的统计信息，以便在会话结束时绘制图表
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

