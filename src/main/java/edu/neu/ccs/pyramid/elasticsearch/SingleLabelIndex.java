package edu.neu.ccs.pyramid.elasticsearch;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.ImmutableSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.index.query.FilterBuilders;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.node.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by chengli on 10/12/14.
 */
public class SingleLabelIndex extends ESIndex{
    private static final Logger logger = LogManager.getLogger();
    String labelField;
    String extLabelField;


    public String getLabelField() {
        return labelField;
    }

    public int getLabel(String id){
        GetResponse response = client.prepareGet(indexName, documentType, id).setFields(this.labelField)
                .execute()
                .actionGet();
        if (logger.isDebugEnabled()){
            logger.debug("getting label from id "+id+", field "+this.labelField);
        }
        return (int) response.getField(this.labelField).getValue();
    }

    public String getExtLabel(String id){
        GetResponse response = client.prepareGet(indexName, documentType, id).setFields(this.extLabelField)
                .execute()
                .actionGet();
        return (String) response.getField(this.extLabelField).getValue();
    }

    /**
     * use whitespace analyzer
     * @param bodyField
     * @param phrase already stemmed
     * @param slop
     * @param labelField
     * @param label
     * @return
     */
    public SearchResponse matchPhraseForClass(String bodyField, String phrase,
                                              int slop,
                                              String labelField, int label){
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchPhraseQuery(bodyField, phrase)
                                .slop(slop).analyzer("whitespace"),
                        FilterBuilders.termFilter(labelField, label))).
                execute().actionGet();

        //        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());

        return response;
    }

    //todo only match train ids
    public long phraseDFForClass(String bodyField, String phrase,
                                 int slop,
                                 String labelField, int label) {
        SearchResponse response = this.matchPhraseForClass(bodyField,phrase,
                slop,labelField,label);
        return response.getHits().getTotalHits();
    }

    //todo only match train ids
    public long DFForClass(String bodyField, String phrase,
                           MatchQueryBuilder.Operator operator,
                           String labelField, int label) {
        SearchResponse response = this.matchForClass(bodyField,phrase,operator,labelField,label);
        return response.getHits().getTotalHits();
    }

    public SearchResponse matchForClass(String bodyField, String phrase,
                                        MatchQueryBuilder.Operator operator,
                                        String labelField, int label){
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchQuery(bodyField,phrase)
                                .operator(operator).analyzer("whitespace"),
                        FilterBuilders.termFilter(labelField,label))).
                execute().actionGet();

        //        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());

        return response;
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

        private String labelField = "label";
        private String extLabelField = "real_label";


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

        public Builder setLabelField(String labelField) {
            this.labelField = labelField;
            return this;
        }

        public Builder setExtLabelField(String extLabelField) {
            this.extLabelField = extLabelField;
            return this;
        }

        public SingleLabelIndex build() throws Exception {
            boolean legal = (clientType.equals("node"))||(clientType.equals("transport"));
            if (!legal){
                throw new IllegalArgumentException("clientType = node or transport");
            }
            SingleLabelIndex esIndex = new SingleLabelIndex();
            esIndex.indexName = indexName;
            esIndex.labelField = labelField;
            esIndex.extLabelField = extLabelField;
            esIndex.documentType = documentType;
            esIndex.clientType = clientType;
            esIndex.clusterName = clusterName;
            esIndex.bodyField = bodyField;

            if (clientType.equals("node")){
                /**
                 * don't hold data
                 */
                Node node = nodeBuilder().client(true).
                        clusterName(clusterName).node();
                esIndex.node = node;
                esIndex.client = node.client();
            } else {
                Settings settings = ImmutableSettings.settingsBuilder()
                        .put("cluster.name", clusterName).build();

                esIndex.client = new TransportClient(settings);
                for (int i=0;i<this.hosts.size();i++){
                    ((TransportClient)esIndex.client)
                            .addTransportAddress(new InetSocketTransportAddress(this.hosts.get(i),
                                    this.ports.get(i)));
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
