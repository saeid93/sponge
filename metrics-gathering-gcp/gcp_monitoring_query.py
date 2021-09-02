"""
    Wrapper for querying monitoring results.


    An example is provided below
    >>> mq = MonitoringQuery(project_id='project_name')
    metrics = [
        "container:cores",
        "container:cores",
        "container:cpu_usage",
        "container:mem_limit",
        "container:mem_limit_usage",
        "container:mem_request",
        "container:mem_request_usage",
        "container:mem_used_bytes",
        "container:uptime"
    ]
    namespace = 'default'
    resource_labels = ['pod_name','container_name']
    df = mq.get_metric(metrics=metrics,)

"""
import datetime
import numpy as np
import logging
import pandas as pd

from typing import List
from dateutil import parser
from google.cloud.monitoring_v3 import MetricServiceClient, query

# ================================================================
#           Wrapper
# ================================================================


class MonitoringQuery:
    # ============================================================
    # For a complete source of metrics, see:
    # https://cloud.google.com/monitoring/api/metrics_kubernetes
    # ============================================================
    RESOURCES = ["container", "node", "pod"]

    RESOURCE_MAPPING = {
        "container": {
            "cores": "kubernetes.io/container/cpu/request_cores",
            "cpu_usage": "kubernetes.io/container/cpu/request_utilization",
            "mem_limit": "kubernetes.io/container/memory/limit_bytes",
            "mem_limit_usage":
            "kubernetes.io/container/memory/limit_utilization",
            "mem_request": "kubernetes.io/container/memory/request_bytes",
            "mem_request_usage":
            "kubernetes.io/container/memory/request_utilization",
            "mem_used_bytes": "kubernetes.io/container/memory/used_bytes",
            "restart": "kubernetes.io/container/restart_count",
            "uptime": "kubernetes.io/container/uptime",
        },
        "node": {
            "alloc_util": "kubernetes.io/node/cpu/allocatable_utilization",
            "core_usagetime": "kubernetes.io/node/cpu/core_usage_time",
            "total_cores": "kubernetes.io/node/cpu/total_cores",
            "mem_allocatable": "kubernetes.io/node/memory/allocatable_bytes",
            "mem_alloc_util":
            "kubernetes.io/node/memory/allocatable_utilization",
            "mem_total_bytes": "kubernetes.io/node/memory/total_bytes",
            "mem_used_bytes": "kubernetes.io/node/memory/used_bytes",
            "network_received":
            "kubernetes.io/node/network/received_bytes_count",
            "network_sent": "kubernetes.io/node/network/sent_bytes_count",
        },
        "pod": {
            "network_received":
            "kubernetes.io/pod/network/received_bytes_count",
            "network_sent": "kubernetes.io/pod/network/sent_bytes_count",
        },
    }

    def __init__(self, project_id: str, logger: object = None):
        self.project_id = project_id
        self.client = MetricServiceClient()
        self.logger = (
            logger if logger is not None else logging.getLogger("__Resource__")
        )
        self.namespace = None

    @property
    def resource_type(self):
        return MonitoringQuery._resources

    @property
    def mappings(self):
        return [
            f"{y}:{x}"
            for y in MonitoringQuery.RESOURCES
            for x in list(MonitoringQuery.RESOURCE_MAPPING[y].keys())
        ]

    def get_metric_mapping(self, spec: str):
        assert spec in self.mappings, "Choose one of: {}".format(
            ",".join(self.mappings)
        )
        keys = spec.split(":")
        return MonitoringQuery.RESOURCE_MAPPING[keys[0]][keys[1]]

    def get_metrics(
        self,
        metrics: str,
        date_from: str = None,
        date_to: str = None,
        days_back: float = 1,
        resource_labels: List[str] = ["resource_type", "pod_name"],
        namespace: str = "default",
        align_type: str = "ALIGN_MAX",
        align_period: int = 1,
        freq: str = "3H",
    ):
        """
        # see e.g. https://stackoverflow.com/questions/63279239/ ...
        #          clarification-riegarding-difference-between-align-mean-and-align-sum-in-google-cl
        """
        self.namespace = namespace
        if not isinstance(metrics, list):
            metrics = [metrics, ]
        _metrics = [self.get_metric_mapping(x) for x in metrics]
        # get basic query
        date_to = (
            parser.parse(
                date_to) if date_to is not None else datetime.datetime.utcnow()
        )
        date_from = (
            parser.parse(date_from)
            if date_from is not None
            else date_to - datetime.timedelta(days=days_back)
        )
        assert date_from < date_to
        #
        _ranges = [f"{x}Z" for x in pd.date_range(date_from, date_to,
                   freq=freq)]
        ranges = [
            (parser.parse(_ranges[i]), parser.parse(_ranges[i + 1]))
            for i in range(len(_ranges) - 1)
        ]
        queries = [
            self._get_query(
                x, namespace, y[0], y[1], align_type, align_period, days_back
            )
            for x in _metrics
            for y in ranges
        ]
        # add labels to the query and exec
        # note this will be slow, in order to speed up recomend to chunk it
        # and throw it in with any concurrent tricks you prefer
        # for now i just limit
        dfs = [
            self._get_result(x, y, resource_labels)
            for x, y in zip(queries, *[metrics * len(ranges)])
        ]
        # process results
        merged = pd.concat(dfs)
        # dropna
        merged = merged.dropna(axis="rows", thresh=1)
        return merged

    def _get_result(self, query_obj: object, resource, labels: List[str]):
        """Execute a query and Returns the result.

        The resource labels will be applied to resulting dataframe, hence
        the columns are MultiIndex.
        """
        df = query_obj.as_dataframe(labels=labels)
        df.columns = df.columns.to_flat_index().map(lambda x: "_".join(x))
        # unstack because the result consists of columns, each
        # correspond to a node/container
        df = df.unstack()
        df = df.reset_index()
        # put back label, we replace the resource concated indentifier with
        # name
        df.columns = ["name", "date", resource]
        df = df.set_index(["name", "date"])
        return df

    def _get_query(
        self,
        metric: str,
        namespace: str,
        date_from: str,
        date_to: str,
        align_type: str,
        align_period: str,
        days: int,
    ):
        """Return the monitoring query object.

        A query is an object consists of some targets resources plus
        various filters.
        See e.g.
        """
        q = query.Query(
            client=self.client, project=self.project_id, days=days,
            metric_type=metric
        )
        # filtering by resources
        q = q.select_resources(namespace_name=namespace)
        # add interval
        q = q.select_interval(end_time=date_to, start_time=date_from)
        # add alignment
        q = q.align(align_type, minutes=align_period)
        return q

    def _process_result_df(self, df):
        """ Process resulting df by quantising using e.g. date """
        self.logger.info("Transform df to docs")
        _df = pd.DataFrame.copy(df)
        self.logger.info(f"Dropped NaN, Remain shape: {_df.shape}")
        # only until minute
        _df.reset_index(inplace=True)
        _df.date = pd.to_datetime(_df.date).dt.round("min")
        _df.date = pd.to_datetime(_df.date).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        _df.set_index(["name", "date"], inplace=True)
        _df.sort_index(inplace=True)
        _df = _df.groupby(["name", "date"]).apply(max).reset_index()
        self.logger.info("Cleaning up ...")
        _df = _df.replace({np.nan: None})
        _df.rename(columns={"date": "timestamp"})
        return _df

    def to_docs(self, result_df):
        """ Format result df """
        _df = pd.DataFrame.copy(result_df)
        _df = self._process_result_df(_df)
        _df["project_id"] = self.project_id
        _df["namespace"] = self.namespace
        return _df.to_dict(orient="records")

# =================================================================================
# Code for dealing with ElasticSearch which is unnecessary for now
# =================================================================================
    # def index_docs_to_es(
    #     self,
    #     docs: List[dict],
    #     es_url: List[str],
    #     es_key: List[str],
    #     index: str = "dev-temp-resource",
    #     es_bulk: str = "parallel_bulk",
    #     chunk_size: int = 500,
    #     timeout: int = 540,
    # ):
    #     """ Index docs to es """
    #     self.logger.info("Indexing data, total: {} docs".format(len(docs)))
    #     if not isinstance(es_url, list):
    #         es_url = [
    #             es_url,
    #         ]
    #     # create
    #     es = Elasticsearch(
    #         es_url,
    #         api_key=es_key,
    #         timeout=timeout,
    #         max_retries=2,
    #         retry_on_timeout=True,
    #     )

    #     actions = []
    #     for doc in docs:
    #         name, date = doc["name"], doc["date"]
    #         _id = f"{name}_{date}"
    #         action = {"_index": index, "_id": _id, "_source": doc}
    #         actions.append(action)
    #     return self._insert(
    #         es_client=es, actions=actions,
    #         method=es_bulk, chunk_size=chunk_size
    #     )

    # def _insert(
    #     self,
    #     es_client: object,
    #     actions: List[Dict],
    #     method: str,
    #     chunk_size: int = 1000,
    #     thread_count: int = 16,
    #     queue_size: int = 16,
    #     request_timeout: int = 600,
    # ):
    #     if method == "bulk":
    #         return helpers.bulk(es_client, actions)
    #     elif method == "parallel_bulk":
    #         data_iter = (x for x in actions)
    #         pb = helpers.parallel_bulk(
    #             client=es_client,
    #             actions=data_iter,
    #             chunk_size=chunk_size,
    #             thread_count=thread_count,
    #             queue_size=queue_size,
    #             request_timeout=request_timeout,
    #         )
    #         deque(pb, maxlen=0)
    #         return "Success!"
    #     else:
    #         raise NotImplementedError()
