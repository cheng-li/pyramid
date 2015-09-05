package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.blog_gender.IndexBuilder;
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
 * index blog gender
 * Created by chengli on 12/26/14.
 */
public class Exp44 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File file = new File(config.getString("input.file"));


        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;

        try(BufferedReader br = new BufferedReader(new FileReader(file))
        ){
            String line;
            while((line=br.readLine())!=null){
                System.out.println("id="+id);
                XContentBuilder builder = IndexBuilder.getBuilder(file, line, id);
                IndexResponse response = client.prepareIndex("blog_gender", "document",""+id)
                        .setSource(builder)
                        .execute()
                        .actionGet();
                id += 1;
            }
        }


        node.close();
    }
}
