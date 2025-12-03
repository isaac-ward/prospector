import matplotlib as mpl

def get_agent_color_list(num_agents: int) -> list:
    """
    Get a list of distinct colors for agents.

    Parameters
    ----------
    num_agents : int
        Number of agents to get colors for.

    Returns
    -------
    list
        List of colors for the agents.
    """
    assert num_agents < 10, "num_agents must be less than 10 for distinct colors."
    return mpl.pyplot.get_cmap("tab10").colors[0:num_agents]