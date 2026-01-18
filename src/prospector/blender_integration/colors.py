# src/prospector/blender_integration/colors.py

def get_agent_color_list(num_agents: int) -> list:
    """
    Get a list of distinct colors for agents. Based off matplotlib tab10 colors.

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
    
    cols = [
        [4, 89, 147],
        [219, 96, 0],
        [17, 128, 17],
        [180, 12, 13],
        [117, 73, 156],
        [109, 57, 46],
        [192, 89, 161],
        [96, 96, 96],
        [155, 156, 7],
        [0, 157, 173],
    ]
    # Into the range 0-1
    return [ [c / 255.0 for c in cols[i]] for i in range(num_agents) ]