package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.imdb.MultiLabelIndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.File;
import java.util.List;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index imdb as multi-label
 * Created by chengli on 2/4/15.
 */
public class Exp65 {
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
            if (MultiLabelIndexBuilder.acceptFile(file)){
                System.out.println("id = "+id);
                XContentBuilder builder = MultiLabelIndexBuilder.getBuilder(file);
                //               System.out.println(builder.string());
                IndexResponse response = client.prepareIndex("imdb_multi_label", "document",""+id)
                        .setSource(builder)
                        .execute()
                        .actionGet();
                id += 1;
            }

        }

        node.close();
    }
}
