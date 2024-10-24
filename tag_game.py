import random, sys
import numpy as np
from collections import deque
from sklearn import preprocessing
import torch
from PG import *
from misc import *

class game_space:
    def __init__(self, dirname, width, height, num_agents, num_teams, walls, bonuses, split_layers=False, flatten_state=False, visible=5, prev_states=4):

        self.dirname = dirname
        self.split_layers = split_layers
        self.flatten_state = flatten_state
        self.visible = visible
        self.previous_states = prev_states
        self.width = width + (2 * self.visible)
        self.height = height + (2 * self.visible)
        self.num_agents = num_agents
        self.max_bonus = bonuses
        self.sep_left = 0.5
        self.sep_right = 0.5
        self.num_teams = num_teams
        self.fam_meet = 0
        self.oth_meet = 0

        #############################################################		
		# Num_actions: 12											#
        # left-yes 	= 0		left-negotiate	= 1		left-no  = 2	#
        # right-yes = 3		right-negotiate	= 4		right-no = 5	#
		# up-yes 	= 6		up-negotiate	= 7		up-no	 = 8	#
        # down-yes  = 9		down-negotiate	= 10	down-no  = 11	#
        #############################################################			

        self.num_actions = 12
        self.walls = walls
        self.num_teams = num_teams
        self.reset()

    def reset(self):
        space = self.make_empty_game_space()

        if self.walls > 0:
            space = self.add_walls(space)
        self.initial_game_space = np.array(space)
        self.game_space = np.array(self.initial_game_space)
        self.create_new_agents()
        self.make_bonuses()
        self.update_agent_positions()

    def open_file(self, dirname, file):

        list = []
        filename = os.path.join(dirname, file)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    list = json.loads(line)
                    break
    
        return list

    def save_file(self, dirname, file, list):
    
        filename = os.path.join(dirname, file)
        with open(filename, "w") as f:
            f.write(json.dumps(list))


    def create_new_agents(self):

        agents_saved = self.open_file(self.dirname, "agents.json")

        if len(agents_saved) > 0:

            self.agents = agents_saved
            self.agent_states = []

            state_size = self.get_state_size()

            for idx, item in enumerate(self.agents):
			
                self.agent_states.append(deque())
                for n in range(self.previous_states):
                    self.agent_states[idx].append(np.zeros(state_size, dtype=int))
                    self.game_space[item[1]][item[0]] = idx+5

        else:

            self.agents = []
            self.agent_states = []

            list_agents = range(self.num_agents)
            teams = [list_agents[i:i + self.num_teams+1] for i in range(0, len(list_agents), self.num_teams+1)]
		
            player_num = 0
            while len(self.agents) < self.num_agents:
                team = 0
                pad = self.visible
			
                gen = self.generate_generosity_rate()

                for i, list in enumerate(teams):
                    if player_num in list:
                        team = i
                        xpos = random.randint(pad, int(self.width-pad)-(pad*2))
                        ypos = random.randint(pad, int(self.height-pad))

                if self.game_space[ypos][xpos] != 0:
                    continue
                overlapped = self.check_for_overlap(xpos, ypos)
                if overlapped == False:
                    self.agents.append([xpos, ypos, team, gen])
                    self.agent_states.append(deque())
                    agent_index = len(self.agents)-1
                    state_size = self.get_state_size()
                    for n in range(self.previous_states):
                        self.agent_states[agent_index].append(np.zeros(state_size, dtype=int))
                
                    self.game_space[ypos][xpos] = team+5

                    player_num += 1

    def make_empty_game_space(self):
        space = np.zeros((self.height, self.width), dtype=int)
        for n in range(self.width):
            for m in range(self.visible):
                space[m][n] = 1
                space[self.height-(m+1)][n] = 1
        for n in range(self.height):
            for m in range(self.visible):
                space[n][m] = 1
                space[n][self.width-(m+1)] = 1
        return space

    def find_random_empty_cell(self, space):
        xpos = 0
        ypos = 0
        empty = False
        pad = self.visible
        while empty == False:
            xpos = random.randint(pad, self.width-(pad*2))
            ypos = random.randint(pad, self.height-(pad*2))
            if space[ypos][xpos] == 0:
                empty = True
                break
        return xpos, ypos

    def add_walls(self, space):
        added = 0
        target = int((self.width * self.height) * self.walls)
        while added < target:
            xpos, ypos = self.find_random_empty_cell(space)
            space[ypos][xpos] = 1
            for n in range(50):
                move = random.randint(0,3)
                if move == 0:
                    xpos = max(0, xpos-1)
                elif move == 1:
                    xpos = min(self.width-1, xpos+1)
                elif move == 2:
                    ypos = max(0, ypos-1)
                elif move == 3:
                    ypos = min(self.height-1, ypos+1)
                if space[ypos][xpos] == 0:
                    added += 1
                space[ypos][xpos] = 1
                if added >= target:
                    break
        return space

    def make_bonuses(self):
        self.bonuses = []
        while len(self.bonuses) < self.max_bonus:
            x, y = self.find_random_empty_cell(self.game_space)
            self.bonuses.append([x, y])
            self.game_space[y][x] = 4

    def add_bonuses(self, space):
        for item in self.bonuses:
            x, y = item
            space[y][x] = 4
        return space

    def replace_bonus(self, xpos, ypos):
        for index, item in enumerate(self.bonuses):
            x, y = item
            if x == xpos and y == ypos:
                newx, newy = self.find_random_empty_cell(self.game_space)
                self.bonuses[index] = [newx, newy]

    def add_agents(self, space):
        for index, item in enumerate(self.agents):
            x, y, s, g = item
            space[y][x] = index+5

        return space

    def update_agent_positions(self):
        space = np.array(self.initial_game_space)
        space = self.add_agents(space)
        space = self.add_bonuses(space)
        self.game_space = space

    def get_visible(self, ypos, xpos):
        lpad = self.visible
        rpad = self.visible+1
        left = xpos-lpad
        right = xpos+rpad
        top = ypos-lpad
        bottom = ypos+rpad
        visible = np.array(self.game_space[left:right, top:bottom], dtype=int)
        return visible

    def get_state_size(self):
        state_size = ((self.visible*2)+1)*((self.visible*2)+1)
        if self.split_layers == True:
            return 4 * state_size
        return state_size

    def get_agent_state(self, index, reputation_saved):
        agent_states = self.agent_states[index]
        x, y, s, g = self.agents[index]
        visible = self.get_visible(x, y)
        state = np.ravel([visible])

        # Call colab agent state to insert values for family and reputation
        ###################################################################

        state = self.create_state_colab(state, index, s, reputation_saved)
        state = np.ravel(state)

        agent_states.append(state)

        if len(agent_states) < self.previous_states:
            for m in range(self.previous_states - len(agent_states)):
                agent_states.append(state)
        if len(agent_states) > self.previous_states:
            agent_states.popleft()
        states = list(agent_states)
        self.agent_states[index] = deque(states)
        states = np.array(states)
        
        return states

    def get_team_stats(self, bonus_rewards, coop_rewards, services, negotiations):
        
        list = []		
		
        for num in range(self.num_teams):
            bon = 0
            coop = 0
            serv = 0
            neg = 0
            for id, item in enumerate(self.agents):
                x, y, s, g = item
                if s == num:
                    bon += bonus_rewards[id]
                    coop += coop_rewards[id]
                    serv += services[id]
                    neg += negotiations[id]
            list.append([bon, coop, serv, neg])
                
        return list

    def get_adjacent(self, ind):
        adjacents = {}
		
        for i in range(self.num_teams):
            adjacents[i] = []

        xpos, ypos, stat, gen = self.agents[ind]
        for index, item in enumerate(self.agents):
            x, y, s, g = item
            
            # left
            if xpos-1 == x and ypos == y:
                adjacents[s].append(index)
            # above left
            if xpos-1 == x and ypos-1 == y:
                adjacents[s].append(index)
            # below left
            if xpos-1 == x and ypos+1 == y:
                adjacents[s].append(index)
            # right
            if xpos+1 == x and ypos == y:
                adjacents[s].append(index)
            # above right
            if xpos+1 == x and ypos-1 == y:
                adjacents[s].append(index)
            # below right
            if xpos+1 == x and ypos+1 == y:
                adjacents[s].append(index)
            # above
            if ypos-1 == y and xpos == x:
                adjacents[s].append(index)
            # below
            if ypos+1 == y and xpos == x:
                adjacents[s].append(index)

        self.get_meetings(adjacents, stat)

        return adjacents

    def get_meetings(self, adjacents, stat):

        for key, value in adjacents.items():
            if stat == key:
                self.fam_meet += len(value)
            else:
                self.oth_meet += len(value)   

    def generate_generosity_rate(self):

		# Number indicates rate of generosity
		# The probability of smaller generosity should be larger
        ########################################################
		
        float_array = np.linspace(0.61, 0.99, num=39)
        list = np.around(float_array, 2).tolist()
        weights = list[::-1]
        gen = random.choices(list, weights=weights, k=1)
        generosity = np.array(gen).item()
        		
        return generosity

    def generate_expectation(self):

        # Expectation id number out of 20
        ##################################

        exp_id = random.choice(range(20))

		# Number indicates intensity of expectation
		# This include importance, urgency, effort, ...
		# The probability of smaller intensity should be larger
        #######################################################
		
        float_array = np.linspace(0.01, 0.99, num=99)
        list = np.around(float_array, 2).tolist()
        weights = list[::-1]
        exp = random.choices(list, weights=weights, k=1)
        expectation = np.array(exp).item()
        		
        return expectation, exp_id
		
    def negotiate(self):

        # Numbers indicate random issue of negotiation
		# Between 0.6 and 0.99 for both parties
        ######################################
		
        float_array = np.linspace(0.6, 0.99, num=40)
        list = np.around(float_array, 2).tolist()

        agent = random.choice(list)
        expectant = random.choice(list)

        return agent, expectant

    def create_state_colab(self, state, index, s, reputation_saved):
		
        # Create state from family relation and agent reputation
        # state = self.get_agent_state(index)
        ####################################
        
        colab_state = []
		
        for item in state:
            if item == 0:
                colab_state.append(item)
            elif item == 4:
                colab_state.append(item)
            elif item == 1:
                colab_state.append(item)
            else:
                item = item - 5				

                if self.agents[item][2] == s:
                    colab_state.append(2)
                else:
                    colab_state.append(3)
					
        return colab_state

    def get_custom_list(self, custom_tableau):

        custom_list = []
        list = []

        # For each kind of expectation and for each agent
        # - determine whether this agent accepted this expectation
        # - determine how many times the agen accepted this expectation
        # - determine whether this agent refused this expectation
        # - determine how many times the agen refused this expectation
        # Create list: expectation id, number of agents saying yes, 
        # total number of yes's, number of agents saying no, 
        # total number of no's		

        for id, exp in enumerate(custom_tableau):

            yes = 0
            num_yes = 0
            no = 0
            num_no = 0
            seuil = 1.0

            for agent in exp:

                if agent[1] != 0:

                    if  agent[0]/agent[1] > seuil:
                        yes += 1 # Number of agent accepting
                        num_yes += agent[0] # Number of acceptances
                    else:
                        no += 1 # number of agents denying
                        num_no += agent[1] # number of denials

                elif agent[1] == 0 and agent[0] != 0:
                        yes += 1 # Number of agent accepting
                        num_yes += agent[0] # Number of acceptances
				
            ag = [id, yes, num_yes, no, num_no]
            list.append(ag)

        # Create custom list
        # Customs have to be uniform, extensive and representative
        # Three types of custom: strong, middle or weak.
        # In order to be a custom it has to respond to three criteria
        # 1. The number of agents saying yes should be above a certain level
        # 2. The number of agents saying no should be below a certain level
        # 3. The ratio between the total number of yes and the total
        #    number of no should be above a certain level
        ##########################################################
        
		
        for expl in list:

            cus = []

            if expl[4] != 0:
                if expl[1] > 10 and expl[3] < 5 and (expl[2]/expl[4] > 1.5):
                    cus.append(3)
                elif expl[1] > 7 and expl[3] < 7 and (expl[2]/expl[4] > 1.2):
                    cus.append(2)
                elif expl[1] > 4 and expl[3] < 9 and (expl[2]/expl[4] > 1.0):
                    cus.append(1)
                else:
                    cus.append(0)
					
            elif expl[4] == 0 and expl[2] != 0:
                if expl[1] > 10 and expl[3] < 5 and expl[2] > 500:
                    cus.append(3)
                elif expl[1] > 7 and expl[3] < 7 and expl[2] > 200:
                    cus.append(2)
                elif expl[1] > 4 and expl[3] < 9 and expl[2] > 50:
                    cus.append(1)
                else:
                    cus.append(0)
            else:
                cus.append(0)

            custom_list.append([expl[0], max(cus)])

        return custom_list
		
    def get_custom_discount(self, exp_id, custom_list):

        discount = 0
		
        for cust in custom_list:
            if cust[0] == exp_id:
                if cust[1] == 1:
                    discount = 0.05
                elif cust[1] == 2:
                    discount = 0.1
                elif cust[1] == 3:					
                    discount = 0.15

        return discount
		
    def get_family_discount(self, exp_id, s_agent):
		
        discount = 0
		
        for index, item in enumerate(self.agents):
            x, y, s, g = item
            if index == exp_id and s == s_agent:
                discount = 0.1
		
        return discount
        
    def change_team_status(self, actions, reputation_saved, custom_tableau):

        rep_sum = sum(reputation_saved)
		
        no_meet = []
		
        if rep_sum == 0:
            reputation_saved  = np.ones(self.num_agents, dtype=float)

        reputation_saved = np.array(reputation_saved)

        rep_norm = preprocessing.normalize([reputation_saved])

        rewards = np.zeros(self.num_agents, dtype=float)
        services = np.zeros(self.num_agents, dtype=float)		
        negotiations = np.zeros(self.num_agents, dtype=float)		
		
        reputation = np.zeros(self.num_agents, dtype=float)

        custom_list = self.get_custom_list(custom_tableau)
        
        for index, item in enumerate(self.agents):
            x, y, s, g = item
			
            adjacents = self.get_adjacent(index)
			
            decision = actions[index]
            		
            for key, list in adjacents.items():
                if len(list) > 0: 
                    for expectant in list:

                        expectation, exp_id = self.generate_expectation()
                        

#######################################################################################
#                          Calculate rewards	                                      #
#######################################################################################
# 	               |        0,3,6,9        |    1,4,7,10    |         2,5,8,11        #
#                  |         = Yes	       |   = Negotiate  |          = No           #
#######################################################################################
# Reward expectant |          exp          |  exp = neg()   |            0            #
#######################################################################################
# Reward agent 	   | exp*gen*(1+discount)* | agent = neg()  |  -0.2*(1+discount)*     #
#                  | (1+fam_discount)*     |                |  (1+fam_discount)*      #
#                  | reputation_expectant  |                |  reputation_expectant   #
#######################################################################################
# Reputation agent | exp*(1+discount)*     |       0        |  -1*exp*(1+discount)*   #
#                  | (1+fam_discount)      |                |  (1+fam_discount)       #
#######################################################################################

                        neg_agent, neg_expectant = self.negotiate()
						
                        discount = self.get_custom_discount(exp_id, custom_list)
                        fam_discount = self.get_family_discount(exp_id, s)
						
                        if decision == 0 or decision == 3 or decision == 6 or decision == 9: 
                        # yes
                            custom_tableau[exp_id][index][0] += 1

							# Expectation between 0.01 and 0.99
                            ###################################

                            services[expectant-5] += expectation

                            # g is the generosity rate (between0.61 and 0.99) 
							# and express the degree to which the agent find
                            # find satisfaction in helping others
                            #####################################

                            rewards[index] += expectation*g*(1+discount)*(1+fam_discount)*rep_norm[0][expectant-5]
                            reputation[index] += expectation*(1+discount)*(1+fam_discount)

							# Custom discount either 0.05, 0.1 or 0.15
							# Family discount 0.1 
                            #####################

                        elif decision == 1 or decision == 4 or decision == 7 or decision == 10:
                        # neg
                            # Negociation rewards random between 0.4 and 
                            # 0.5 for both parties multiplied with expectation.
                            ###################################################

                            negotiations[expectant-5] += neg_expectant*expectation
                            negotiations[index] += neg_agent*expectation
                            reputation[index] += 0

                        elif decision == 2 or decision == 5 or decision == 8 or decision == 11:
                        # no
                            custom_tableau[exp_id][index][1] += 1

                            services[expectant-5] += 0

							# Custom discount either 0.05, 0.1 or 0.15
							# Family discount 0.1 
                            #####################
							
                            rewards[index] += -0.2*(1+discount)*(1+fam_discount)* rep_norm[0][expectant-5]
                            reputation[index] += -1*expectation*(1+discount)*(1+fam_discount)


        return rewards, services, negotiations, reputation, custom_tableau, custom_list

    def move_agent(self, index, move):
        reward = 0
        x, y, s, g = self.agents[index]
        newx = x
        newy = y
        if move == 0 or move == 1 or move == 2: # left
            newx = max(0, x-1)
        elif move == 3 or move == 4 or move == 5: # right
            newx = min(self.width-1, x+1)
        elif move == 6 or move == 7 or move == 8: # up
            newy = max(0, y-1)
        elif move == 9 or move == 10 or move == 11: # down
            newy = min(self.height-1, y+1)

        moved = False
        if newx != x or newy != y:
            item = self.game_space[newy][newx]
            if item == 0:
                moved = True
            elif item == 4:
                moved = True
                reward = 2.5   # Bonus reward
                self.replace_bonus(newx, newy)

        if moved == True:
            if self.check_for_overlap(newx, newy) == True:
                moved = False

        if moved == True:
            x = newx
            y = newy
        self.agents[index] = [x, y, s, g]
        return reward

    def check_for_overlap(self, xpos, ypos):
        for item in self.agents:
            x, y, s, g = item
            if xpos == x and ypos == y:
                return True
        return False

    def get_printable(self, item):
        if item == 0:
            return "\x1b[1;37;47m" + "  " + "\x1b[0m"
        elif item == 1:	
            return "\x1b[1;35;40m" + "░░" + "\x1b[0m"
        elif item == 4:
            return "\x1b[1;31;47m" + "© " + "\x1b[0m"
        elif item > 4:
            
				
            for index, item_ag in enumerate(self.agents):
                x, y, s, g = item_ag
                if item-5 == index:
                    if item < 10:
                        item = " " + str(item)
                    color = 30+s
                    return "\x1b[1;47;" + str(color) + "m" + str(item) + "\x1b[0m"
            

    def print_game_space(self):
        printable = ""
        pad = self.visible - 1
        for column in self.game_space[pad:-pad]:
            for item in column[pad:-pad]:
                printable += self.get_printable(item)
            printable += "\n"
        return printable

