package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.reuters.IndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.node.Node;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by Rainicy on 8/2/15.
 */
public class Exp205 {
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the conifg file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String clusterName = config.getString("index.clusterName");
        String indexName = config.getString("index.name");
        String indexType = config.getString("index.type");

        String mekaData = config.getString("meka.data");
        int numLabels = config.getInt("number.labels");


        Node node = nodeBuilder().client(true).clusterName(clusterName).node();
        Client client = node.client();

        List<String> attributes = new ArrayList<>();
        int id = 0;
        String line = "";
        Boolean ifData = false;

        BufferedReader br = new BufferedReader(new FileReader(mekaData));
        while ((line = br.readLine()) != null) {
            if (line.startsWith("@attribute")) {
                String attribute = line.split(" ")[1];
                attributes.add(attribute);
            } else if (line.startsWith("@data")) {
                ifData = true;
            } else if (ifData) {
                String[] splitInfo = line.split(",");
                List<String> labels = new ArrayList<>();
                String body = "";
                for (int i=0; i<splitInfo.length; i++) {
                    String elem = splitInfo[i];
                    // labels
                    if (i < numLabels) {
                        if (elem.equals("1")) {
                            labels.add(attributes.get(i));
                        }
                    } else {
                        body += elem.substring(1, elem.length()-1);
                    }
                }
                if (!body.equals("")) {
                    System.out.println("id: " + id);
                    XContentBuilder builder = getBuilder(labels, body, id);
                    IndexResponse response = client.prepareIndex(indexName, indexType, ""+id)
                            .setSource(builder)
                            .execute()
                            .actionGet();
                    ++id;
                }
            }
        }
        br.close();
        node.close();
    }

    private static XContentBuilder getBuilder(List<String> labels, String body, int id) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body", body);
        builder.array("labels", labels.toArray(new String[labels.size()]));
        builder.field("split", ((id%5) != 0) ? "train" : "test");
        builder.endObject();
        return builder;
    }
}
