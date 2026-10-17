from hou_memory import Memory

class MultiAgentMemoryStore:
    """
    Intermediary between agents and the memory databases
    Operations are done asynchronously
    """

    __init__(self):
        pass

    def add_memory(
        self,
        agent,
        memory: Memory,
        ):
        """
        Parameters:
            agent: the agent that made this request
            memory: the memory to add
        """
        pass

    def recall_memories(
        self,
        agent,
        **kwargs: dict[str, str],
        ):
        """
        Parameters:
            agent: the agent that made this request
            kwargs: optional arguments; columns to include
                top_k: Number of scores to retrieve
        """
        pass

    def reinforce_memories(
        self,
        agent,
        memory_ids: list[str],
        ):
        """
        Parameters:
            agent: the agent that made this request
            memory_ids: list of memories to reinforce
        """
        pass
