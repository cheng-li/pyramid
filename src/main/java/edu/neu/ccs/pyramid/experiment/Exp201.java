package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.reuters.IndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index reuters 2004.
 * Created by Rainicy on 6/26/15.
 */
public class Exp201 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the conifg file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String dir = config.getString("input.folder");
        String logs = config.getString("output.log");

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();

        List<File> files = DirWalker.getFiles(dir);
        int id = 0;
        int missingId = 1;
        BufferedWriter bw = new BufferedWriter(new FileWriter(logs));
        for (File file : files) {
            if (file.getName().endsWith("xml")) {
                System.out.println("id = " + id);
                try {
                    XContentBuilder builder = IndexBuilder.getBuilder(file);
                    IndexResponse response = client.prepareIndex(config.getString("index.name"),
                            config.getString("index.type"), ""+id)
                            .setSource(builder)
                            .execute()
                            .actionGet();
                    ++id;
                } catch (NullPointerException e) {
                    System.out.println("null topic : " + file.getName());
                    bw.write(missingId + ": " + file.toString()+ "\n");
                    ++missingId;
//                    e.printStackTrace();
                    continue;
                }
            }
        }
        bw.close();
        node.close();
    }
}
