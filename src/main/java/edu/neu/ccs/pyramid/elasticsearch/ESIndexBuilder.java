package edu.neu.ccs.pyramid.elasticsearch;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.ImmutableSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.node.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by chengli on 9/5/14.
 */
public class ESIndexBuilder {
    private String indexName = "unknown_index";
    private String labelField = "label";
    private String extLabelField = "real_label";
    private String multiLabelField = "multi_label";
    private String extMultiLabelField ="real_multi_label";
    private String documentType = "document";
    private String clientType = "transport";
    private String clusterName = "elasticsearch";
    private String bodyField = "body";
    private List<String> hosts = new ArrayList<>();
    private List<Integer> ports = new ArrayList<>();
    private int termVectorCacheSize = 10000;

    public static ESIndexBuilder builder(){
        return new ESIndexBuilder();
    }

    public ESIndexBuilder setIndexName(String indexName) {
        this.indexName = indexName;
        return this;
    }

    public ESIndexBuilder setLabelField(String labelField) {
        this.labelField = labelField;
        return this;
    }

    public ESIndexBuilder setExtLabelField(String extLabelField) {
        this.extLabelField = extLabelField;
        return this;
    }

    public ESIndexBuilder setDocumentType(String documentType) {
        this.documentType = documentType;
        return this;
    }

    public ESIndexBuilder setClientType(String clientType) {
        this.clientType = clientType;
        return this;
    }

    public ESIndexBuilder setClusterName(String clusterName) {
        this.clusterName = clusterName;
        return this;
    }

    public ESIndexBuilder addHostAndPort(String host, int port){
        this.hosts.add(host);
        this.ports.add(port);
        return this;
    }

    public ESIndexBuilder addHostsAndPorts(String[] hosts, String[] ports){
        for (int i=0;i< hosts.length;i++){
            addHostAndPort(hosts[i],Integer.parseInt(ports[i]));
        }
        return this;
    }

    public ESIndexBuilder setBodyField(String bodyField) {
        this.bodyField = bodyField;
        return this;
    }

    public ESIndexBuilder setTermVectorCacheSize(int termVectorCacheSize) {
        this.termVectorCacheSize = termVectorCacheSize;
        return this;
    }

    public ESIndexBuilder setMultiLabelField(String multiLabelField) {
        this.multiLabelField = multiLabelField;
        return this;
    }

    public ESIndexBuilder setExtMultiLabelField(String extMultiLabelField) {
        this.extMultiLabelField = extMultiLabelField;
        return this;
    }

    public ESIndex build() throws Exception {
        boolean legal = (clientType.equals("node"))||(clientType.equals("transport"));
        if (!legal){
            throw new IllegalArgumentException("clientType = node or transport");
        }
        ESIndex esIndex = new ESIndex();
        esIndex.indexName = indexName;
        esIndex.labelField = labelField;
        esIndex.extLabelField = extLabelField;
        esIndex.documentType = documentType;
        esIndex.clientType = clientType;
        esIndex.clusterName = clusterName;
        esIndex.bodyField = bodyField;
        esIndex.multiLabelField = multiLabelField;
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
