from .cluster import Cluster # noqa
from gym.envs.registration import register # noqa

register(
    id='SimEnv-v0',
    entry_point='smart_kube.envs:SimEnv',
)
register(
    id='KubeEnv-v0',
    entry_point='smart.kube.envs:KubeEnv',
)
