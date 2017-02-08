package edu.neu.ccs.pyramid.elasticsearch;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.index.query.Operator;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.node.Node;
import org.elasticsearch.transport.client.PreBuiltTransportClient;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;

/**
 * Created by chengli on 10/12/14.
 */
public class MultiLabelIndex extends ESIndex{
    private static final Logger logger = LogManager.getLogger();
    String extMultiLabelField;

    public List<String> getExtMultiLabel(String id){
        return getStringListField(id,this.extMultiLabelField);
    }


    public SearchResponse matchForClass(String phrase, String extLabel){
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setTrackScores(false).
                setFetchSource(false).setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.boolQuery()
                		.should(QueryBuilders.matchQuery(bodyField, phrase)
                				.operator(Operator.AND).analyzer("whitespace"))
                		.must(QueryBuilders.matchQuery(this.extMultiLabelField, extLabel)))
                .execute().actionGet();

        //        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());

        return response;
    }

    public long DFForClass(String phrase, String extLabel) {
        SearchResponse response = this.matchForClass(phrase,extLabel);
        return response.getHits().getTotalHits();
    }





    public static class Builder {
        private String indexName = "unknown_index";
        private String documentType = "document";
        private String clientType = "transport";
        private String clusterName = "elasticsearch";
        private String bodyField = "body";
        private List<String> hosts = new ArrayList<>();
        private List<Integer> ports = new ArrayList<>();
        private int termVectorCacheSize = 10000;

        private String multiLabelField = "multi_label";
        private String extMultiLabelField ="real_multi_label";


        public Builder setIndexName(String indexName) {
            this.indexName = indexName;
            return this;
        }

        public Builder setDocumentType(String documentType) {
            this.documentType = documentType;
            return this;
        }

        public Builder setClientType(String clientType) {
            this.clientType = clientType;
            return this;
        }

        public Builder setClusterName(String clusterName) {
            this.clusterName = clusterName;
            return this;
        }

        public Builder addHostAndPort(String host, int port){
            this.hosts.add(host);
            this.ports.add(port);
            return this;
        }

        public Builder addHostsAndPorts(String[] hosts, String[] ports){
            for (int i=0;i< hosts.length;i++){
                addHostAndPort(hosts[i],Integer.parseInt(ports[i]));
            }
            return this;
        }

        public Builder setBodyField(String bodyField) {
            this.bodyField = bodyField;
            return this;
        }

        public Builder setTermVectorCacheSize(int termVectorCacheSize) {
            this.termVectorCacheSize = termVectorCacheSize;
            return this;
        }

        public Builder setMultiLabelField(String multiLabelField) {
            this.multiLabelField = multiLabelField;
            return this;
        }

        public Builder setExtMultiLabelField(String extMultiLabelField) {
            this.extMultiLabelField = extMultiLabelField;
            return this;
        }



        public MultiLabelIndex build() throws Exception {
            boolean legal = (clientType.equals("node"))||(clientType.equals("transport"));
            if (!legal){
                throw new IllegalArgumentException("clientType = node or transport");
            }
            MultiLabelIndex esIndex = new MultiLabelIndex();
            esIndex.indexName = indexName;

            esIndex.documentType = documentType;
            esIndex.clientType = clientType;
            esIndex.clusterName = clusterName;
            esIndex.bodyField = bodyField;
            esIndex.extMultiLabelField = extMultiLabelField;

            if (clientType.equals("node")){
                /**
                 * don't hold data
                 */
            	
            	Settings settings = Settings.builder()
                        .put("cluster.name", clusterName)
                        .put("node.data", false)
                        .build();
            	
                esIndex.client = new PreBuiltTransportClient(settings);
                //TODO Check ?
                ((TransportClient)esIndex.client)
                .addTransportAddress(new InetSocketTransportAddress(new InetSocketAddress("127.0.0.1",
                        9300)));
            } else {
                Settings settings = Settings.builder()
                        .put("cluster.name", clusterName).build();

                esIndex.client = new PreBuiltTransportClient(settings);
                for (int i=0;i<this.hosts.size();i++){
                    ((TransportClient)esIndex.client)
                            .addTransportAddress(new InetSocketTransportAddress(new InetSocketAddress(this.hosts.get(i),
                                    this.ports.get(i))));
                }
            }
            esIndex.numDocs = esIndex.fetchNumDocs();

            esIndex.termVectorCache = CacheBuilder.newBuilder()
                    .maximumSize(this.termVectorCacheSize)
                    .build(new CacheLoader<String, Map<Integer, String>>() {
                        @Override
                        public Map<Integer, String> load(String id) throws Exception {
                            return esIndex.getTermVectorFromIndex(id);
                        }
                    });

            return esIndex;
        }

    }

}
