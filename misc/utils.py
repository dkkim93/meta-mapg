import yaml
import logging
import git
import numpy as np


def load_config(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters

    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to load config from. Default: "."
    """
    with open(path + "/config/" + args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
    """Sets python logging

    Args:
        logger_name (str): Specifies logging name
        log_file (str): Specifies path to save logging
        level (int): Logging when above specified level. Default: logging.INFO
    """
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters

    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to get Git repository. Default: "."

    Examples:
        log[args.log_name].info("Hello {}".format("world"))

    Returns:
        log (dict): Dictionary that contains python logging
    """
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./log/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    repo = git.Repo(path)
    log[args.log_name].info("Branch: {}".format(repo.active_branch))
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    return log


def to_onehot(value, dim):
    """Convert batch of numbers to onehot

    Args:
        value (numpy.ndarray): Batch of numbers to convert to onehot. Shape: (batch,)
        dim (int): Dimension of onehot

    Returns:
        onehot (numpy.ndarray): Converted onehot. Shape: (batch, dim)
    """
    assert len(value.shape) == 1, "Shape must be (batch,)"
    onehot = np.eye(dim, dtype=np.float32)[value]
    assert onehot.shape == (value.shape[0], dim), "Shape must be: (batch, dim)"
    return onehot


def log_performance(scores, log, tb_writer, args, i_joint, iteration, rank, is_train=True):
    """Log performance of training at each task

    Args:
        scores (list): Contains scores for each agent
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        i_joint (int): Joint policy index in Markov chain
        iteration (int): Iteration of training during meta-train, meta-val, or meta-test
        rank (int): Index of process for multi-threading
        is_train (bool): Flag to identify whether in meta-train or not
    """
    for i_agent, score in enumerate(scores):
        if args.reward_scale:
            score *= args.ep_horizon * 100

        log[args.log_name].info(
            "[Rank::{}] At iteration {}, reward: {:.5f} for agent {} at joint policy {}".format(
                rank, iteration, score, i_agent, i_joint))

        if is_train:
            key1 = "rank" + str(rank) + "/reward"
            key2 = "agent" + str(i_agent) + "/joint" + str(i_joint)
            tb_writer.add_scalars(key1, {key2: score}, iteration)
