import numpy as np
import random
from maze import CogintiveMaze
import matplotlib.pyplot as plt

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
        
        """actr-r chunk holder"""
        self.chunk_holder = self.init_chunk_holder()
        
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
        R = np.matrix([[-100,0,0,-100,-100], 
                       [-100,-100,0,0,-100], 
                       [-100,0,-100,0,100], 
                       [-100,0,0,-100,-100], 
                       [-100,-100,0,-100,100]])
        return R
    
    def init_q_table(self, matrix_size = 5):
        """
        define q table: 
        """
        Q = np.matrix(np.zeros([matrix_size, matrix_size]))
        return Q
    
    def init_chunk_holder(self):
        return {"VISUAL":[], "GOAL":[], "IMAGINAL":[], "RETRIEVAL":[], "MOTOR":[]}
    
        
    ##########################################
    ### TODO:  define cognitive actions
    ##########################################    
    def perform_actions(self, current_state, action):
        print('in perform_actions', current_state, action)
        if current_state == 3:
            self.retrieve_memory()
            self.copy_chunks(current_state, action)
        elif current_state == 4:
            self.press_button()
        else:
            self.copy_chunks(current_state, action)
        
    def copy_chunks(self, from_buffer, to_buffer):
        chunk = self.chunk_holder[from_buffer] 
        self.chunk_holder[to_buffer] = chunk
        print('COPY CHUNK:"{}" \nFROM {} TO -> {}'.format(chunk,  
                                                          self.maze.mappings[from_buffer], 
                                                          self.maze.mappings[to_buffer]))
    
    def retrieve_memory(self):
        '''chunk = np.random.choice(["LEFT", "RIGHT"], p=[1, 0])
        if chunk == self.chunk:
            # update reward table
            self.r_table[1:3, 3] = 20
        else:
            self.r_table[1:3, 3] = 0
        return chunk'''
        
        """
        This simple random retrieval method rerturn retrieved memory based on probability
        Futher implementation should add ACT-R memory  retrieval
        """
        print('RETRIEVING MEMORY ...')
        retrieved_chunk = np.random.choice(['LEFT', 'RIGHT'], p=[0.9, 0.1])
        self.chunk_holder['RETRIEVAL'] = retrieved_chunk
        
    def press_button(self):
        print('>> RESPOND', self.chunk_holder['MOTOR'])

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
    
    def update(self, current_state, action, gamma):
        """
        update q table based by exploring the enviroment
        """
        print('in update', current_state, action)
        Q = self.q_table
        R = self.r_table
        
        current_state = self.maze.mappings[current_state]
        action = self.maze.mappings[action]
        
        #current_state = int(self.maze.mappings[current_state])
        #action = int(self.maze.mappings[action])
        
        max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size = 1))
        else:
            max_index = int(max_index)
        max_value = Q[action, max_index]

        Q[current_state, action] = R[current_state, action] + gamma * max_value
        # print('max_value', R[current_state, action] + gamma * max_value)
        
        # perform action
        self.perform_actions(current_state, action)
        
        # update q table
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
            
            #self.available_states(current_state)

            next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

            if next_step_index.shape[0] > 1:
                next_step_index = int(np.random.choice(next_step_index, size = 1))
            else:
                next_step_index = int(next_step_index)

            steps.append(next_step_index)
            current_state = next_step_index

        print("Learned most efficient path:")
        print([self.maze.nodes[s] for s in steps])