import torch
import numpy as np
from meta.base import Base
from misc.rl_utils import get_entropy_loss, get_gae
from misc.torch_utils import get_parameters, zero_grad, ensure_shared_grads
from torch.nn.utils.convert_parameters import parameters_to_vector as to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters as to_parameters
from meta.projection_grad import pc_grad
from misc.shared_adam import SharedAdam


class MetaAgent(Base):
    """Class for training a meta-agent
    Args:
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
        rank (int): Used for thread-specific meta-agent for multiprocessing. Default: -1
    """
    def __init__(self, log, tb_writer, args, name, i_agent, rank=-1):
        super(MetaAgent, self).__init__(log, tb_writer, args, name, i_agent, rank)

        self._set_dim()
        self._set_action_type()
        self._set_linear_baseline()
        self._set_policy()
        self._set_dynamic_lr()

    def _set_policy(self):
        if self.is_discrete_action:
            from network.categorical_lstm import ActorNetwork, ValueNetwork
            self.log[self.args.log_name].info("[{}] Set Categorical LSTM policy".format(self.name))
        else:
            from network.gaussian_lstm import ActorNetwork, ValueNetwork
            self.log[self.args.log_name].info("[{}] Set Gaussian LSTM policy".format(self.name))

        self.actor = ActorNetwork(self.input_dim, self.output_dim, self.name, self.args)
        self.log[self.args.log_name].info("[{}] {}".format(self.name, self.actor))

        self.value = ValueNetwork(self.input_dim, self.name, self.args)
        self.log[self.args.log_name].info("[{}] {}".format(self.name, self.value))

    def _set_dynamic_lr(self):
        initial_lr = np.array([self.args.actor_lr_inner for _ in range(self.args.chain_horizon)])
        self.dynamic_lr = torch.nn.Parameter(torch.from_numpy(initial_lr).float(), requires_grad=True)

    def share_memory(self):
        self.actor.share_memory()
        self.actor_optimizer = SharedAdam(
            get_parameters(self.actor), lr=self.args.actor_lr_outer, amsgrad=True)
        self.actor_optimizer.share_memory()

        self.dynamic_lr.share_memory_()
        self.dynamic_lr_optimizer = SharedAdam(
            get_parameters(self.dynamic_lr), lr=self.args.actor_lr_outer, amsgrad=True)
        self.dynamic_lr_optimizer.share_memory() 

        self.value.share_memory()
        self.value_optimizer = SharedAdam(
            get_parameters(self.value), lr=self.args.value_lr, amsgrad=True)
        self.value_optimizer.share_memory() 

    def sync(self, shared_meta_agent):
        assert shared_meta_agent.rank == -1, "Shared meta-agent's rank must be -1 (i.e., non-thread)"

        self.actor.load_state_dict(shared_meta_agent.actor.state_dict())
        self.dynamic_lr.data = shared_meta_agent.dynamic_lr.data
        self.value.load_state_dict(shared_meta_agent.value.state_dict())

    def _get_outer_actor_loss(self, memories, iteration):
        # Get advantage
        _, _, _, values_phi, rewards_phi = memories[-1].sample()
        advantage_phi = get_gae(values_phi[self.i_agent], rewards_phi[self.i_agent], self.args).detach()

        # Get base loss
        _, logprobs_theta, _, _, _ = memories[0].sample()
        self_logprob_theta = torch.stack(logprobs_theta[self.i_agent], dim=1)
        base_loss = torch.mean(torch.sum(self_logprob_theta * advantage_phi, dim=1))

        # Get self-shaping loss
        self_shaping_loss = 0.
        for i_memory, memory in enumerate(memories[1:]):
            _, logprobs, _, _, _ = memory.sample()
            self_logprob = torch.stack(logprobs[self.i_agent], dim=1)
            self_shaping_loss += torch.mean(torch.sum(self_logprob * advantage_phi, dim=1))

        # Get opponent-shaping loss
        opponent_shaping_loss = 0.
        if self.args.opponent_shaping:
            i_agent = 1  # NOTE Assuming only two agents with [meta_agent, opponent]
            for i_memory, memory in enumerate(memories[1:]):
                _, logprobs, _, _, _ = memory.sample()
                opponent_logprob = torch.stack(logprobs[i_agent], dim=1)
                opponent_shaping_loss += torch.mean(torch.sum(opponent_logprob * advantage_phi, dim=1))

        # For logging
        key = "rank" + str(self.rank) + "/outer/actor_loss"
        self.tb_writer.add_scalars(key, {"base": -base_loss}, iteration)
        self.tb_writer.add_scalars(key, {"self_shaping": -self_shaping_loss}, iteration)
        self.tb_writer.add_scalars(key, {"opponent_shaping": -opponent_shaping_loss}, iteration)

        # Return final loss
        return -(base_loss + self_shaping_loss + opponent_shaping_loss)

    def get_outer_loss(self, memories, iteration):
        if len(memories) == 1:
            entropy_loss = get_entropy_loss(memories[0], self.args, self.i_agent)
            value_loss = 0.
            return entropy_loss, value_loss

        # Compute actor loss
        actor_loss = self._get_outer_actor_loss(memories, iteration)

        # Compute baseline loss
        _, _, _, values_phi, rewards_phi = memories[-1].sample()
        value_loss = self._get_value_loss(values_phi[self.i_agent], rewards_phi[self.i_agent])

        return actor_loss, value_loss

    def outer_update(self, shared_meta_agent, loss, process_dict, iteration, update_type):
        loss = sum(loss) / float(len(loss))

        if update_type == "actor":
            network, shared_network = self.actor, shared_meta_agent.actor
            shared_optimizer = shared_meta_agent.actor_optimizer
            key, tb_key = "/actor_grad", "rank" + str(self.rank) + "/outer/actor_loss_avg"
        elif update_type == "dynamic_lr":
            network, shared_network = self.dynamic_lr, shared_meta_agent.dynamic_lr
            shared_optimizer = shared_meta_agent.dynamic_lr_optimizer
            key, tb_key = "/dynamic_lr_grad", None
        elif update_type == "value":
            network, shared_network = self.value, shared_meta_agent.value
            shared_optimizer = shared_meta_agent.value_optimizer
            key, tb_key = "/value_grad", "rank" + str(self.rank) + "/outer/value_loss_avg"
        else:
            raise ValueError()

        zero_grad(network)
        loss.backward(retain_graph=(update_type == "actor"))
        torch.nn.utils.clip_grad_norm_(get_parameters(network), self.args.max_grad_clip)

        # Apply projection conflicting gradient
        process_dict[str(self.rank) + key] = \
            np.copy(to_vector([param.grad for param in get_parameters(network)]).detach().numpy())
        projected_grad = pc_grad(process_dict, self.rank, self.args, key)
        to_parameters(torch.from_numpy(projected_grad), [param._grad for param in get_parameters(network)])

        # Update networks
        ensure_shared_grads(network, shared_network)
        shared_optimizer.step()

        # For logging
        if tb_key is not None:
            self.tb_writer.add_scalars(tb_key, {"agent" + str(self.i_agent): loss.data.numpy()}, iteration)
