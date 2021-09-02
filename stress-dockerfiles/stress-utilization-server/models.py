import numpy as np


class Service:
    """Service Model"""

    def __init__(self, hostname: str, is_up: bool = False):
        """Service

        :param hostname: str
            hostname of service

        :param is_up: bool (default: False)
            service is up or not
        """
        self.is_up: bool = is_up
        self.hostname: str = hostname

    def __str__(self):
        return "Service(hostname='{}', up='{}')".format(
            self.hostname, self.is_up
        )


class Resources:
    """Resources Model"""

    def __init__(self, cpu: str, ram: str):
        """Resources

        :param ram: str
            RAM usage of service

        :param cpu: str
            cpu load of service
        """
        self.ram: str = ram
        self.cpu: str = cpu

    def __str__(self):
        return "Resources(RAM='{}', CPU='{}')".format(
            self.ram, self.cpu
        )


class WorkLoads:
    """WorkLoads Model"""

    def __init__(self, data: np.array):
        """WorkLoads

        :param data: np.array (nResources x nTimesteps)
            all of workloads (it a 2D matrix)
            1D: number of resources
            2D: number of timesteps
        """
        self.data: np.array = np.array(data)
        self.nResources, self.nTimesteps = self.data.shape

    def get_resources(self, timestep: int) -> np.array:
        """Get Resorces

        :param timestep: int
            refers to timestep

        :return: np.array
            nResources x 1
        """
        return self.data[:, timestep]

    def __str__(self):
        return "WorkLoads(nResources='{}', nTimesteps='{}')".format(
            *self.data.shape
        )


class Dataset:
    """Dataset Model"""

    def __init__(self, data: dict):
        self.data = data
