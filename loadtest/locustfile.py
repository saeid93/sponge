from __future__ import annotations

import logging
from typing import Any

from locust import HttpUser, event
from locustfiles.saeed_model import Task1


class PrimaryUser(HttpUser):
    tasks = (Task1)


@events.quitting.add_listener
def _(environment, **kwargs: Any) -> None:
    print(type(environment))
    if environment.stats.total.fail_ratio > 0:
        logging.error("Test failed due to failure ratio > 1%")
        environment.process_exit_code = 1
