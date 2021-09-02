from .base import Base


class Azure(Base):
    def __init__(self, config, timesteps):
        super().__init__(config, timesteps)
