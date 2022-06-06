import numpy as np
import random
from maze import CogintiveMaze
import matplotlib.pyplot as plt
import networkx as nx

##########################
#  Agent 
#########################

class Agent:
    def __init__(self) -> None:
        """init environment"""
        self.maze = CogintiveMaze()
        
        """init agent """
        self.gamma = 0.8  # learning parameter
        self.q_table = self.init_q_table()
        self.r_table = self.init_r_table()
        
        self.state = 0
        self.pi = None
        self.action = 0
        self.state_history = list() #[[0, np.nan]]
        self.step_log = list()
        self.q_table_history = list()
        self.index = 0
        
        self.chunk_holder = self.init_chunk_holder()
        self.update_reward = False
        
    def action_space(self):
        """
        0 = copy chunk 0
        1 = copy chunk 1
        """
        actions = [0, 1]
        return actions
    
    def state_space(self):
        """
        0 = VISUAL MODULE
        1 = GOAL MODULE
        2 = IMAGINAL MODULE
        3 = RETRIEVAL MODULE
        4 = MOTOR MODULE
        """
        states = self.maze.nodes
    
    def init_r_table(self, matrix_size = 5):
        """
        define reward table
        """
        #R = np.matrix([[-100,0,0,-100,-100], 
        #               [-100,-100,0,0,-100], 
        #               [-100,0,-100,0,100], 
        #               [-100,0,0,-100,-100], 
        #               [-100,-100,0,-100,100]])
        R = np.matrix([[-100,0,0,0,0], 
                      [-100,-100,0,0,-100], 
                      [-100,0,-100,0,100], 
                      [-100,0,100,-100,-100], 
                      [-100,-100,0,-100,100]])
        return R
    
    def init_q_table(self, matrix_size = 5):
        """
        define q table: m x n
        """
        Q = np.matrix(np.zeros([matrix_size, matrix_size]))
        return Q
    
    def init_chunk_holder(self, chunk = {'S1':'LEFT'}):
        return {"VISUAL":chunk, "GOAL":None, "IMAGINAL":None, "RETRIEVAL":None, "MOTOR":None}
    

    ##########################################
    ### TODO:  define cognitive actions
    ##########################################    
    def perform_actions(self, current_state, action):
        #print('in perform_actions', current_state, action)
        if current_state == 3:
            self.copy_chunks(current_state, action)
            self.retrieve_memory()
            
        elif current_state == 4:
            self.press_button()
        else:
            self.copy_chunks(current_state, action)
        
    def copy_chunks(self, current_state, action, verbose=False):
        from_buffer = self.maze.nodes[current_state]
        to_buffer = self.maze.nodes[action]
        chunk = self.chunk_holder[from_buffer] 
        self.chunk_holder[to_buffer] = chunk
        if verbose: print('COPY CHUNK:"{}" \nFROM {} TO -> {}'.format(chunk,  
                                                          self.maze.mappings[from_buffer], 
                                                          self.maze.mappings[to_buffer]))
    
    def retrieve_memory(self):
        """
        This simple random retrieval method rerturn retrieved memory based on probability
        Futher implementation should add ACT-R memory  retrieval
        """
       
        chunk = {"S1" : np.random.choice(['LEFT', 'RIGHT'], p=[0, 1])}
        #print('RETRIEVING MEMORY ...', chunk)
        #print('BEFORE: ', self.chunk_holder)
        self.chunk_holder['RETRIEVAL'] = chunk
        #print('AFTER: ', self.chunk_holder)
        return chunk
        
    def press_button(self, v = False):
        if v: print('>> RESPOND', self.chunk_holder['MOTOR'])
        if self.update_reward: self.update_r_table()
        
    def update_r_table(self):
        # if not respond - reward becomes 0
        if self.chunk_holder['MOTOR'] == None:
            self.r_table[2, 4] = 0
            self.r_table[4, 4] = 0
        # if respond - reward becomes 100
        elif self.chunk_holder['MOTOR']['S1'] == 'LEFT':
            self.r_table[2, 4] = -100
            self.r_table[4, 4] = -100
        else:
            self.r_table[2, 4] = 100
            self.r_table[4, 4] = 100
        
        
        
    ##########################################
    ### define actions and states
    ##########################################
    def available_states(self, state, v=False):        
        av_states = [b for (a,b) in self.maze.edges if a==state]
        if v: 
            print("AVAILABLE STATES: {} -> {}".format(self.maze.mappings[state], [self.maze.mappings[a] for a in av_states]))
        return av_states

    def sample_next_state(self, current_state):
        """
        Random pick next reachable actions. If at the corner 4, start from 0
        """
        av_states = self.available_states(state = current_state)
        if len(av_states) > 0:
            next_state = np.random.choice(av_states)
        else: 
            next_state = current_state #self.maze.nodes[0]
        return next_state

    def update_q(Q, state, state2, reward, action, action2):
        predict = Q[state, action]
        target = reward + gamma * Q[state2, action2]
        Q[state, action] = Q[state, action] + alpha * (target - predict)
        return Q
    
    def get_max_state(self, Q, current_state, steps, greedy=0):
        """
        not completely gready, by allowing some exploration behavior
        """
        #print('in get_max_state', current_state)
        
        av_states = self.available_states(self.maze.nodes[current_state])
        av_states_id = [self.maze.mappings[s] for s in av_states]
        
        # if in retrieval loop, directualy GO TO MOTOR
        if self.check_loops(steps):
            av_states_id = [4]
        return np.where(Q[current_state,] == np.max(Q[current_state, av_states_id]))[1]
        '''
        if random.random() > greedy:
            return np.where(Q[action,] == np.max(Q[action,]))[1]
        else:
            return np.array([np.random.choice(np.where(Q[action,] < np.max(Q[action,]))[1])])
        '''
    
    def check_loops(self, steps):
        try:
            
            step1, step2, step3, step4 = steps[-1], steps[-2], steps[-3], steps[-4]
            if step1==step3 and step2==step4:
                return True
            else:
                return False
        except:
            return False
        
    def update(self, current_state, action, gamma):
        """
        update q table based by exploring the enviroment
        """
        Q = self.q_table
        R = self.r_table
        
        current_state = self.maze.mappings[current_state]
        action = self.maze.mappings[action]
        
        max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size = 1))
        else:
            max_index = int(max_index)
        max_value = Q[action, max_index]

        Q[current_state, action] = R[current_state, action] + gamma * max_value
        # print('max_value', R[current_state, action] + gamma * max_value)
        
        self.perform_actions(current_state, action)
        self.q_table = Q

        if (np.max(Q) > 0):
            return(np.sum(Q/np.max(Q)*100))
        else:
            return (0)
        
    def q_learning(self, n=100, v=False):
        # Training
        scores = []
        for i in range(n):
            current_state = np.random.choice(self.maze.nodes)
            next_state = self.sample_next_state(current_state)
            score = self.update(current_state, next_state, self.gamma)
            scores.append(score)
            if v: print ('Score:', str(score))
            
        if v:
            print("Trained Q matrix:")
            print(self.q_table/np.max(self.q_table)*100)
            
        plt.plot(scores)
        plt.show()
        
   
        
            
    def q_testing(self):
        Q = self.q_table
        
        current_state  = self.maze.mappings['VISUAL']
        steps = [current_state]

        while current_state != self.maze.mappings['MOTOR']:                
            #next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
            next_step_index = self.get_max_state(Q, current_state, steps=steps)

            if next_step_index.shape[0] > 1:
                next_step_index = int(np.random.choice(next_step_index, size = 1))
            else:
                next_step_index = int(next_step_index)
            #print('next step id', self.maze.nodes[next_step_index])

            steps.append(next_step_index)
            current_state = next_step_index

        print("Learned most efficient path:")
        print([self.maze.nodes[s] for s in steps])
        
        
    def matrix2dict(self):
        mat = self.q_table
        dic = {}
        for e in self.maze.edges:
            dic[e] = np.round(mat[self.maze.mappings[e[0]], self.maze.mappings[e[1]]], 2)
        return dic

        
   