package edu.neu.ccs.pyramid.elasticsearch;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.ImmutableSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.index.query.FilterBuilders;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.node.Node;

import java.util.*;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

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
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchQuery(bodyField, phrase)
                                .operator(MatchQueryBuilder.Operator.AND).analyzer("whitespace"),
                        FilterBuilders.termFilter(this.extMultiLabelField, extLabel))).
                execute().actionGet();

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

    public LabelTranslator loadLabelTranslator() throws Exception{

        int numDocs = this.getNumDocs();
        Set<String> extLabelSet = new HashSet<>();

        for (int i=0;i<numDocs;i++){
            List<String> extLabel = this.getExtMultiLabel(""+i);
            extLabelSet.addAll(extLabel);
        }

        LabelTranslator labelTranslator = new LabelTranslator(extLabelSet);

//            System.out.println("there are "+labelTranslator.getNumClasses()+" classes.");
//            System.out.println(labelTranslator);
        return labelTranslator;
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
