package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.cnn.IndexBuilder;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.ESIndexBuilder;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;


/**
 * build cnn index
 * Created by chengli on 9/13/14.
 */
public class Exp5 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        String file = config.getString("input.file");

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        try(BufferedReader br = new BufferedReader(new FileReader(new File(file)))){
            String line = null;
            int id = 0;
            while((line=br.readLine())!=null){
                if (IndexBuilder. acceptLine(line)){
                    XContentBuilder builder = IndexBuilder.getBuilder(line);
                    IndexResponse response = client.prepareIndex("cnn", "document",""+id)
                            .setSource(builder)
                            .execute()
                            .actionGet();
                    id += 1;
                }
            }
        }
        node.close();
    }
}
