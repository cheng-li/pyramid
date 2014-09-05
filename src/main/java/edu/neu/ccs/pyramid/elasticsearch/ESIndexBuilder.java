package edu.neu.ccs.pyramid.elasticsearch;

import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.ImmutableSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.node.Node;

import java.util.ArrayList;
import java.util.List;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by chengli on 9/5/14.
 */
public class ESIndexBuilder {
    private String indexName = "unknown_index";
    private String labelField = "label";
    private String extLabelField = "real_label";
    private String type = "document";
    private String clientType = "transport";
    private String clusterName = "elasticsearch";
    private List<String> hosts = new ArrayList<>();
    private List<Integer> ports = new ArrayList<>();

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

    public ESIndexBuilder setType(String type) {
        this.type = type;
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

    public ESIndex build() throws Exception {
        boolean legal = (clientType.equals("node"))||(clientType.equals("transport"));
        if (!legal){
            throw new IllegalArgumentException("clientType = node or transport");
        }
        ESIndex esIndex = new ESIndex();
        esIndex.indexName = indexName;
        esIndex.labelField = labelField;
        esIndex.extLabelField = extLabelField;
        esIndex.type = type;
        esIndex.clientType = clientType;
        esIndex.clusterName = clusterName;

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

        return esIndex;
    }

}
