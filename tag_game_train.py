from tag_game import *
from misc import *
import time, sys, os, re
from collections import Counter

game_space_width = 30 # 60
game_space_height = 20
preds_len = 20
num_agents = 20
walls = 0.03
bonuses = 20
episode_limit = 1000
model_type = "PG"
num_teams = 4
prev_states = 8  #num_frames

for epoque in range(1):

    # Initialize variables
    ######################

    total_rewards = 0
    episode = 1
    temp = 0
    current_iterations = 0
    iteration = 0
    rep_list = []
	
    # Construct dirname
    ###################	

    dirname = model_type + "_tag_game_save"

    # Get indices of all currently saved models
    ###########################################

    trained_models = []
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        fns = os.listdir(dirname)
        for fn in fns:
            m = re.match(".+\_([0-9]+)\.pt$", fn)
            if m is not None:
                model_num = m.group(1)
                trained_models.append(int(model_num))

    # Get gamespace
    ###############
    
    gs = game_space(dirname, game_space_width, game_space_height, num_agents, num_teams, walls, bonuses)

    # Open files
    ############

    tracked_rewards = gs.open_file(dirname, "rewards.json")

    tracked_decisions = gs.open_file(dirname, "decisions.json")

    tracked_bonuses = gs.open_file(dirname, "bonus.json")

    tracked_coops = gs.open_file(dirname, "coop.json")

    tracked_reputations = gs.open_file(dirname, "reputation.json")

    custom_tableau = gs.open_file(dirname, "custom.json")

    tracked_customs = gs.open_file(dirname, "custom_lists.json")

    tracked_fam_stats = gs.open_file(dirname, "fam_stats.json")

    tracked_meetings = gs.open_file(dirname, "meetings.json")

    tracked_services = gs.open_file(dirname, "services.json")

    tracked_negotiations = gs.open_file(dirname, "negotiations.json")

    if not custom_tableau :
        custom_tableau = np.zeros((20, num_agents, 2), dtype = np.int32).tolist()

    if len(tracked_rewards) > 0:
        for list in tracked_rewards:
            total_rewards += sum(list) 
			        
        episode = len(tracked_rewards)

    total_tracked_rewards = total_rewards
	
    if len(tracked_reputations) > 0 :
        reputation_saved = tracked_reputations[-1]
    else:
        reputation_saved = np.zeros(num_agents, dtype=float).tolist()

    # Get models
    ############

    models = []
    for n in range(gs.num_agents):
        model = get_model(model_type, gs, reputation_saved)
        models.append(model)

    # While_loop for episode
	########################

    while True:

        # Initialize variables
        ######################

        iteration += 1
        current_iterations += 1
        done = 0

        if current_iterations >= episode_limit:
            done = 1

        states = []
        actions = []
        rewards = []
        bonus_rewards = []

        # Calculate sum of coop rewards and services rewards
        ####################################################

        if not tracked_coops :
            coop_sum = np.zeros((num_agents), dtype = np.float32)
        elif len(tracked_coops) < 2 :
            coop_sum = tracked_coops[0]
        else:
            coop_sum = [sum(x) for x in zip (*tracked_coops)]
       
        if not tracked_services :
            services_sum = np.zeros((num_agents), dtype = np.float32)
        elif len(tracked_services) < 2 :
            services_sum = tracked_services[0]
        else:
            services_sum = [sum(x) for x in zip (*tracked_services)]
        
        # Rewards from collecting bonuses
        #################################

        for index, agent in enumerate(gs.agents):
            x, y, s, g = gs.agents[index]

            # Determine ratio between moral rewards and services received
            # for agent in question and determine if yes-decision should be blocked
            #######################################################################

            not_yes = False

            if services_sum[index] != 0:
                coop_serv_ratio = coop_sum[index]/services_sum[index]
            else:
                coop_serv_ratio = coop_sum[index]/0.000001

            if coop_serv_ratio > 0.5 * g:
                not_yes = True
			
            # Get pred and probability
            ##########################
			
            state = get_state(gs, index, model_type, reputation_saved)
            states.append(state)
            pred, probs_agent = models[index].get_action(state, g)

            # Block yes-decision if services received too small
            ###################################################
			
            if not_yes == True:
                if pred == 0: pred = 2
                elif pred == 3: pred = 5
                elif pred == 6: pred = 8
                elif pred == 9: pred = 11

            # Get action from pred
            ######################
				
            temp = np.zeros(gs.num_actions, dtype=float)
            temp[pred] = 1.0
            pred = temp
            move = np.argmax(pred)
            actions.append(move)

            # Get bonus reward
            ##################
                 
            reward  = gs.move_agent(index, move)
            bonus_rewards.append(reward)

            # Update agent positions               
            ########################

            gs.update_agent_positions()

        # Cooperative rewards
        #####################

        reputation = []
        rep_cumul = []

        moral_rewards, services, negotiations, reputation, custom_tableau, custom_list = gs.change_team_status(actions, reputation_saved, custom_tableau)


        # Calculate cumulated reputation
        ################################

        rep_cumul = [sum(x) for x in zip(reputation_saved, reputation)]
		
        reputation_saved = rep_cumul

        # Calculate statistics for each family_stats
        ############################################

        family_stats = gs.get_team_stats(bonus_rewards, moral_rewards, services, negotiations)

        # Calculate rewards and total rewards
        #####################################

        rewards = [sum(x) for x in zip(bonus_rewards, moral_rewards, services, negotiations)]

        total_rewards += sum(rewards)

        # Append to tracked files
        #########################

        tracked_rewards.append(rewards)

        actions = [int(x) for x in actions]
        tracked_decisions.append(actions)

        tracked_bonuses.append(bonus_rewards)
        
        tracked_coops.append(moral_rewards.tolist())
        
        tracked_reputations.append(rep_cumul)
        
        tracked_fam_stats.append(family_stats)

        meetings = [gs.fam_meet, gs.oth_meet]
			
        tracked_meetings.append(meetings)

        tracked_services.append(services.tolist())

        tracked_negotiations.append(negotiations.tolist())

        gs.fam_meet = 0
        gs.oth_meet = 0

        # Print visuals
        ###############

        os.system('clear')
            
        msg = "Model: " + model_type
        msg += " Iteration: " + str(iteration)
        msg += " Episode: " + str(episode)
        msg += " Total_tracked_rewards: " + str(total_tracked_rewards)
        msg += " Total rewards: " + "%.2f"%total_rewards
        msg += "\nAgents: " + str(gs.agents)

        print()
        print(gs.print_game_space())
        print(msg)

        time.sleep(0.005)

        # Send info to models (rewards)
        ###############################
			
        for n in range(gs.num_agents):
            models[n].push_replay(rewards[n])

        # If epoque ends
        #################

        if done == 1:

            # Update policy
            ###############

            for n in range(gs.num_agents):
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write("Training model: " + "%03d"%n)
                sys.stdout.flush()
                models[n].update_policy()

            # Save models and files
            #######################

            for n in range(gs.num_agents):
                models[n].save_model(dirname, n)

            total_final_rewards = 0
            
            gs.save_file(dirname, "rewards.json", tracked_rewards)        

            if len(tracked_rewards) > 0:
                for list in tracked_rewards:
                    total_final_rewards += sum(list)

            gs.save_file(dirname, "decisions.json", tracked_decisions)

            gs.save_file(dirname, "bonus.json", tracked_bonuses)

            gs.save_file(dirname, "coop.json", tracked_coops)
            
            gs.save_file(dirname, "reputation.json", tracked_reputations)

            gs.save_file(dirname, "custom.json", custom_tableau)
			
            tracked_customs.append(custom_list)
            
            gs.save_file(dirname, "custom_lists.json", tracked_customs)
            
            gs.save_file(dirname, "fam_stats.json", tracked_fam_stats)

            gs.save_file(dirname, "agents.json", gs.agents)
			
            gs.save_file(dirname, "meetings.json", tracked_meetings)

            gs.save_file(dirname, "services.json", tracked_services)

            gs.save_file(dirname, "negotiations.json", tracked_negotiations)

            # End matters
            #############

            episode += 1

            #last_episode_length = current_iterations
            current_iterations = 0

            gs.reset()

            break
            
            
            



            
            
            

