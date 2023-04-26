import gym
from gym import spaces
import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import PPO

class NimEnv(gym.Env):
    def __init__(self,stones, opp):
        super(NimEnv,self).__init__()
        self.opp = opp
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(stones+1)
        self.n = stones
        self.num_stones = stones
        self.spec = None
        self.metadata = None
   
    def reset(self):
        self.num_stones = self.n
        return self.num_stones
    
    def step(self, action):
        if (action + 1 > self.num_stones):
            return self.num_stones, -10, True, {}
        if (self.num_stones - (action + 1) == 0):
            self.num_stones = 0
            return 0, 1, True, {}
        else:
            new_stones = self.num_stones - (action + 1)
            opp_action = self.opp.move(new_stones)
            if new_stones - opp_action == 0:
                self.num_stones = 0
                return 0, -1, True, {'opp_action' : opp_action}
            else:
                self.num_stones = new_stones - opp_action
                return new_stones - opp_action, 0, False, {'opp_action' : opp_action}
    
    def render(self, mode="console"):
        print(str(self.num_stones) + " stones on the board.")
    
    def close(self):
        pass


class opponent1():
    def __init__(self,y):
        self.y = y
    def move(self,n):
        moves = [1,2,3]
        table = [""]*(n+1) 
        winning = 0
        for spot in range(len(table)):
            if spot < min(moves):
                table[spot] = "L"
            else:
                found = False
                for move in moves:
                    if spot - move == 0:
                        table[spot] = "W"
                        winning = move
                        found = True
                        break
                if not found:
                    found = False
                    for move in moves:
                        if move > spot:
                            continue
                        if table[spot-move] == "L":
                            winning = move
                            found = True
                            break
                    if found:
                        table[spot] = "W"
                    else:
                        table[spot] = "L"
        if table[-1] == "L":
            return random.randint(1,min(n,3))
        else:
            if random.randint(0,100) < self.y:
                return(winning)
            else:
                return random.randint(1,min(n,3)) 

class opponent2():
    def __init__(self,y):
        self.y = y
    def move(self,n):
        y = 30
        moves = [1,2,3]
        table = [""]*(n+1) 
        winning = 0
        for spot in range(len(table)):
            if spot < min(moves):
                table[spot] = "L"
            else:
                found = False
                for move in moves:
                    if spot - move == 0:
                        table[spot] = "W"
                        winning = move
                        found = True
                        break
                if not found:
                    found = False
                    for move in moves:
                        if move > spot:
                            continue
                        if table[spot-move] == "L":
                            winning = move
                            found = True
                            break
                    if found:
                        table[spot] = "W"
                    else:
                        table[spot] = "L"
        if table[-1] == "L":
            return random.randint(1,min(n,3))
        else:
            if random.randint(0,100) < self.y:
                return(winning)
            else:
                return random.randint(1,min(n,3)) 

            


modelTrain = [10,20,30,40,50,60,70,80,90,100]
modelTest = [10,20,30,40,50,60,70,80,90,100]

for train in modelTrain:
    print("TRAIN Y = "+str(train))
    opp1 = opponent1(train)   
    env1 = NimEnv(50, opp1)
    check_env(env1)
    env = make_vec_env(lambda: env1, n_envs = 1)
    model = PPO('MlpPolicy', env, verbose=0).learn(total_timesteps = 25000)
    for m in modelTest:
        print("TEST Y = "+str(m))
        wins = 0
        losses = 0
        misfires = 0
        opp2 = opponent2(m)   
        env2 = NimEnv(50, opp2)
        env = make_vec_env(lambda: env2, n_envs = 1)
        for i in range(10000):
            obs = env.reset()
            finished = False
            while(not finished):
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                if(done):
                    finished = True
                    if reward == 1:
                        wins += 1
                    if reward == -1:
                        losses += 1
                    if reward == -10:
                        misfires += 1
            
        print('Wins: ' + str(wins))
        print('Losses: ' + str(losses))
        print('Invalid action attempts: ' + str(misfires))
        print('Winrate: ' + str((wins/10000)))