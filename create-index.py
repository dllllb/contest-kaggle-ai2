import common

# Running Elasticsearch server is required
# docker run -p 9200:9200 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:5.2.1

common.index_ck_12_conecpts_v8()
