package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;

import edu.neu.ccs.pyramid.data_formatter.twenty_newsgroup.IndexBuilder;
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
 * index 20newsgroup
 * Created by chengli on 11/14/14.
 */
public class Exp23 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        String folder = config.getString("input.folder");
        List<File> files = DirWalker.getFiles(folder);

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;
        for (File file: files){
            System.out.println("id = "+id);
            XContentBuilder builder = IndexBuilder.getBuilder(file,id);
            //               System.out.println(builder.string());
            IndexResponse response = client.prepareIndex("20newsgroup", "document",""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            id += 1;
        }

        node.close();
    }
}
