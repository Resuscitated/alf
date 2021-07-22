class PerProcessContext(object):
    __instance = None
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.rank = 0
        return cls.__instance


    def set_rank(self, rank: int):
        self.rank = rank
