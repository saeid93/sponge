import os
import sys
import pickle
import json
import click
from pprint import PrettyPrinter
from smart_kube.workload import SyntheticWorkloadGenerator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import ( # noqa
    WORKLOADS_PATH,
    CONFIGS_PATH
)


pp = PrettyPrinter(indent=4)


# generaing the workloads
def generate_workload(notes: str,
                      workload_type: str,
                      timesteps: int,
                      time_interval: int,
                      #   plot_smoothing: int,
                      config: dict,
                      container: dict,
                      seed: int
                      ):
    """
        generate a random workload
    """
    # fix foldering per datast
    workloads_path = os.path.join(WORKLOADS_PATH, 'synthetic')
    content = os.listdir(workloads_path)
    new_workload = len(content)
    dir2save = os.path.join(workloads_path, str(new_workload))
    os.mkdir(dir2save)

    # generate the workload
    workload_generator = SyntheticWorkloadGenerator(
        workload_type=workload_type,
        time_interval=time_interval,
        # plot_smoothing,
        seed=seed,
        container=container,
        timesteps=timesteps,
        config=config)
    workload, fig, time = workload_generator.make_workload()

    # information of the generated workload
    info = {
        'workload_type': workload_type,
        'notes': notes,
        'timesteps': timesteps,
        # 'plot_smoothing': plot_smoothing,
        'config': config,
        'seed': seed
    }
    if workload_type in ['constant', 'step', 'sinusoidal', 'lowhigh']:
        info.update({'time_interval': time_interval})

    # save the information and workload in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(dir2save, 'container.json'), 'x') as out_file:
        json.dump(container, out_file, indent=4)
    with open(os.path.join(dir2save, 'workload.pickle'), 'wb') as out_pickle:
        pickle.dump(workload, out_pickle)
    with open(os.path.join(dir2save, 'time.pickle'), 'wb') as out_pickle:
        pickle.dump(time, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")

    # save figs
    fig.savefig(os.path.join(dir2save, 'figure.png'))


@click.command()
@click.option('--workload-type', type=click.Choice(
    ['constant', 'step', 'sinusoidal', 'lowhigh']),
    default='sinusoidal')
def main(workload_type: str):
    print('generating workload from the following workoad type: {}'.format(
        workload_type
    ))
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'workloads',
        f'{workload_type}.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    generate_workload(workload_type=workload_type,
                      **config)
    pp.pprint(config)


if __name__ == "__main__":
    main()
