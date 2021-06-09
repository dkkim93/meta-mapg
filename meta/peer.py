import torch
from meta.base import Base
from misc.torch_utils import change_name


class Peer(Base):
    """Class for training a peer
    Args:
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
        rank (int): Used for thread-specific meta-agent for multiprocessing. Default: -1
    """
    def __init__(self, log, tb_writer, args, name, i_agent, rank=-1):
        super(Peer, self).__init__(log, tb_writer, args, name, i_agent, rank)

        self._set_dim()
        self._set_action_type()
        self._set_linear_baseline()
        self._set_policy()

    def _set_policy(self):
        # For repeated matrix game experiments, we consider tabular representation for 
        # the opponent's policy, which will be directly set when set_persona() is called.
        # Thus, returning instead of setting policy
        if self.args.env_name == "IPD-v0" or self.args.env_name == "RPS-v0":
            self.is_tabular_policy = True
        else:
            if self.is_discrete_action:
                from network.categorical_lstm import ActorNetwork
                self.log[self.args.log_name].info("[{}] Set Categorical LSTM policy".format(self.name))
            else:
                from network.gaussian_lstm import ActorNetwork
                self.log[self.args.log_name].info("[{}] Set Gaussian LSTM policy".format(self.name))

            self.actor = ActorNetwork(self.input_dim, self.output_dim, self.name, self.args)
            self.log[self.args.log_name].info("[{}] {}".format(self.name, self.actor))

    def set_persona(self, persona):
        if self.args.env_name == "IPD-v0" or self.args.env_name == "RPS-v0":
            self.log[self.args.log_name].info("[{}] Set persona: {}".format(self.name, persona))
            self.actor = torch.nn.Parameter(torch.from_numpy(persona).float(), requires_grad=True)
        else:
            self.log[self.args.log_name].info("[{}] Set persona: {}".format(self.name, persona["iteration"]))
            actor = torch.load(persona["filepath"])["actor_state_dict"]
            actor = change_name(actor, old="teammate", new="peer")
            self.actor.load_state_dict(actor)
