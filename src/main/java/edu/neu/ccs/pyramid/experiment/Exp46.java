package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.trec_spam.IndexBuilder;
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
 * index trac spam
 * Created by chengli on 12/28/14.
 */
public class Exp46 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        String folder = config.getString("input.folder");
        List<File> files = DirWalker.getFiles(folder);
        File labelIndexFile = new File(config.getString("input.labelFile"));

        IndexBuilder indexBuilder = new IndexBuilder(labelIndexFile);

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;
        for (File file: files){
//            System.out.println(file.getAbsolutePath());
            System.out.println("id = "+id);
            XContentBuilder builder = indexBuilder.getBuilder(file, id);
            //               System.out.println(builder.string());
            IndexResponse response = client.prepareIndex(config.getString("index.name"), "document",""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            id += 1;
        }
        node.close();
    }
}
