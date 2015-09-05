package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.recon.IndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.List;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index recon
 * Created by chengli on 4/27/15.
 */
public class Exp89 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }


        Config config = new Config(args[0]);
        System.out.println(config);


        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;
        File pos = new File(config.getString("input.posFile"));
        File neg = new File(config.getString("input.negFile"));

        try(BufferedReader br = new BufferedReader(new FileReader(pos))){
            String line = null;
            while((line=br.readLine())!=null){
                XContentBuilder builder = IndexBuilder.getBuilder(line, "pos");
                IndexResponse response = client.prepareIndex("recon", "document",""+id)
                        .setSource(builder)
                        .execute()
                        .actionGet();
                System.out.println(""+id);
                id += 1;
            }
        }


        try(BufferedReader br = new BufferedReader(new FileReader(neg))){
            String line = null;
            while((line=br.readLine())!=null){
                XContentBuilder builder = IndexBuilder.getBuilder(line, "neg");
                IndexResponse response = client.prepareIndex("recon", "document",""+id)
                        .setSource(builder)
                        .execute()
                        .actionGet();
                System.out.println(""+id);
                id += 1;
            }
        }


        node.close();
    }

}
