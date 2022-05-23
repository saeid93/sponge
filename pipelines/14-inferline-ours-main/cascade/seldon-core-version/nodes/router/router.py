import logging
import random
import os


# NUMBER_OF_ROUTES = int(os.environ.get("NUMBER_OF_ROUTES", "2"))


class Router:
    def __init__(self) -> None:
        super().__init__()
        logging.info("Router initialized!")

    def route(self, features, names=[], meta=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")
        logging.info(f"model meta: {meta}")
        # logging.info(f"number of routes: {NUMBER_OF_ROUTES}")
        # route = random.randint(0, NUMBER_OF_ROUTES)
        logging.info(f"routing to: {0}")
        return 0