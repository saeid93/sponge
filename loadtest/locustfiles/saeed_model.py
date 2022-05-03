from __future__ import annotations

from http import HTTPStatus

from locust import TaskSet, between, task

import settings
import setup


class Task1(TaskSet):
    """Converting from Dollar to RS & vice versa."""

    # Time period between firing consecutive tasks is 1-3 seconds
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Logins and stuff before starting a user session."""
        print("Start work")

    @task
    def task_one(self) -> None:
        url = "/api/v0.1/prediction"

        querystring = {"data":[1,2,3]}

        headers = {
            "x-rapidapi-host": settings.HOST,
            "x-rapidapi-key": settings.API_TOKEN,
        }

        with self.client.post(
            url,
            headers=headers,
            params=querystring,
            catch_response=True,
        ) as response:
            if response.status_code == HTTPStatus.OK:
                response.success()
            else:
                response.failure(f"Failed! Http Code `{response.status_code}`")

    @task
    def stop(self) -> None:
        """TaskSet objects don't know when to hand over control
        to the parent class. This method does exactly that."""

        self.interrupt()
