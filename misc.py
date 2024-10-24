import numpy as np
import torch
from PG import *

def get_state(gs, index, model_type, reputation_saved):
    state = torch.FloatTensor(gs.get_agent_state(index, reputation_saved))
    state = state.unsqueeze(0)
    return state

def get_model(model_type, gs, reputation_saved):
    num_actions = gs.num_actions

    num_frames = gs.get_agent_state(0, reputation_saved).shape[0]
    state_size = gs.get_agent_state(0, reputation_saved).shape[1]
    model = PG_model(state_size, num_actions, num_frames)

    return model
