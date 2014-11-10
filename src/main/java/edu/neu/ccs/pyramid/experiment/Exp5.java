package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.cnn.IndexBuilder;
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
        String folder = config.getString("input.folder");
        List<File> files = DirWalker.getFiles(folder);

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;
        for (File file: files){
            try(BufferedReader br = new BufferedReader(new FileReader(file))){
                String line = null;
                while((line=br.readLine())!=null){
                    if (IndexBuilder. acceptLine(line)){
                        XContentBuilder builder = IndexBuilder.getBuilder(file,line);
                        IndexResponse response = client.prepareIndex("cnn", "document",""+id)
                                .setSource(builder)
                                .execute()
                                .actionGet();
                        System.out.println(""+id);
                        id += 1;
                    }
                }
            }
        }

        node.close();
    }
}
