from models import WorkLoads
from flask import jsonify
from flask import Flask
import schedule
import requests
import threading
import logging
import pickle
import time

log_format = '%(asctime)s,%(msecs)d %(levelname)-8s' +\
     ' [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

# Flask configurations
PORT = 80
app = Flask(__name__)

# Time Interval (seconds)
INTERVAL = 60

# WorkLoads
WORKLOAD_PATH = '/workloads.pickle'
WORKLOADS: WorkLoads

# k8s namespace
NAMESPACE = 'vpa'

# Services
SERVICES = dict()


"""NOTE: Don't touch these variables"""
# Enable Scheduler
IS_ENABLED_SCHEDULER = False

# Current TimeStep
CURRENT_TIME_STEP = 0


@app.route('/metrics/<string:hostname>/', methods=['GET'])
def metrics(hostname: str):
    global WORKLOADS, CURRENT_TIME_STEP, SERVICES, \
        IS_ENABLED_SCHEDULER, INTERVAL
    """Register a hostname and return its current metrics

    :param hostname: str
        hostname of container
    """

    # service does not exist
    if SERVICES.get(hostname, None) is None:
        logging.info(
            'service "{}" not exist, init resources are allocated'.format(
                hostname
                ))

        INIT_RAM, INIT_CPU = WORKLOADS.data[:, CURRENT_TIME_STEP]

        service = {
            hostname: {
                "specs": {
                    "ram": INIT_RAM,
                    "cpu": INIT_CPU
                }
            }
        }

        SERVICES.update(service)

        if not IS_ENABLED_SCHEDULER:
            schedule.every(INTERVAL).seconds.do(updateTimesteps)
            threading.Thread(target=runScheduler).start()
            IS_ENABLED_SCHEDULER = True

        return jsonify(SERVICES.get(hostname).get('specs')), 200

    # service exists, moved service exists
    return jsonify(SERVICES.get(hostname).get('specs')), 200


def updateTimesteps():
    global CURRENT_TIME_STEP, SERVICES, NAMESPACE

    logging.info('increasing the current time step from {} to {}'.format(
        CURRENT_TIME_STEP, CURRENT_TIME_STEP + 1
    ))
    CURRENT_TIME_STEP = CURRENT_TIME_STEP % WORKLOADS.nTimesteps + 1

    try:
        for hostname, detail in SERVICES.items():

            URL = "http://{}.{}.svc/".format(hostname, NAMESPACE)

            # fetch current timestep resource usage
            RAM, CPU = WORKLOADS.data[:, CURRENT_TIME_STEP]

            logging.info(
                "Updating specs of service '{}' from '{}' to '{}'".format(
                    hostname,
                    SERVICES[hostname]['specs'],
                    {
                        'ram': RAM,
                        'cpu': CPU
                    }
                    ))

            SERVICES[hostname]['specs'] = {
                'ram': RAM,
                'cpu': CPU
            }

            requests.get(
                url=URL, params=SERVICES[hostname]['specs'], timeout=3)
    except Exception as e:
        logging.error(e)


def runScheduler():

    time.sleep(2)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':

    while True:
        try:

            with open(WORKLOAD_PATH, 'rb') as file:
                WORKLOADS = WorkLoads(pickle.load(file))

            break
        except Exception as e:
            logging.info(
                'looking for file "{}", '
                'you need to upload them: {}'.format(
                    WORKLOAD_PATH,
                    e
                ))
        time.sleep(1)

    logging.info("serving 'app' on port {}".format(PORT))
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
