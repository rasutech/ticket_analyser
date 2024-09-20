# elk.py
from elasticsearch import Elasticsearch

def search_elk_logs(elk_host, start_time, end_time, work_step_identifier):
    """Search ELK logs for errors related to a work step between the given start and end time."""
    es = Elasticsearch([elk_host])
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"work_step_identifier": work_step_identifier}},
                    {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}}
                ]
            }
        }
    }
    res = es.search(index="logs-*", body=query)
    if res['hits']['hits']:
        return res['hits']['hits'][0]['_source']['message']
    return "No logs found for the failed work step"
