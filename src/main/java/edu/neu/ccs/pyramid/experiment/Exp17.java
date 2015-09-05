package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.classic.Classic3IndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index classic3
 * Created by chengli on 10/28/14.
 */
public class Exp17 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String dir = config.getString("input.folder");

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();

        // ignore cacm
        List<File> list = DirWalker.getFiles(dir).stream().filter(file -> !file.getName().startsWith("cacm"))
                .collect(Collectors.toList());
        int id = 0;
        for (File file : list) {
            System.out.println("id = "+id);
            XContentBuilder builder = Classic3IndexBuilder.getBuilder(file);
            //               System.out.println(builder.string());
            IndexResponse response = client.prepareIndex("classic3", "document",""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            id += 1;
        }

        node.close();
    }
}
