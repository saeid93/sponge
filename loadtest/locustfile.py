from __future__ import annotations

import logging
from typing import Any

from locust import HttpUser, event
from locustfiles.saeed_model import Task1

from locust import HttpUser, TaskSet, task, constant
from locust import LoadTestShape

class PrimaryUser(HttpUser):
    tasks = (Task1)



class DoubleWave(LoadTestShape):
    """
    A shape to imitate some specific user behaviour. In this example, midday
    and evening meal times. First peak of users appear at time_limit/3 and
    second peak appears at 2*time_limit/3
    Settings:
        min_users -- minimum users
        peak_one_users -- users in first peak
        peak_two_users -- users in second peak
        time_limit -- total length of test
    """

    min_users = 20
    peak_one_users = 60
    peak_two_users = 40
    time_limit = 10

    def tick(self):
        run_time = round(self.get_run_time())
        print(run_time)
        if run_time < self.time_limit:
            user_count = run_time
            return (10, 3)
        else:
            return None


@events.quitting.add_listener
def _(environment, **kwargs: Any) -> None:
    print(type(environment))
    if environment.stats.total.fail_ratio > 0:
        logging.error("Test failed due to failure ratio > 1%")
        environment.process_exit_code = 1
