package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.ohsumed.IndexBuilder;

import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.logging.ESLoggerFactory;
import org.elasticsearch.common.logging.log4j.Log4jESLoggerFactory;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.File;
import java.io.FileReader;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index Ohsumed_20000
 * Created by chengli on 10/1/14.
 */
public class Exp8 {
    public static void main(String[] args) throws Exception{

        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String dir  = config.getString("input.folder");

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();

        List<File> list = DirWalker.getFiles(dir);
        //make sure each file is indexed only once
        Set<String> added = new HashSet<>();

        Map<String, Set<String>> nameToCodesMap = IndexBuilder.collectCodes(dir);
        int id = 0;
        for (File file: list){
            if (!added.contains(file.getName())){
                System.out.println("id = "+id);
                XContentBuilder builder = IndexBuilder.getBuilder(file,nameToCodesMap);
//               System.out.println(builder.string());
                IndexResponse response = client.prepareIndex("ohsumed_20000", "document",""+id)
                        .setSource(builder)
                        .execute()
                        .actionGet();
                id += 1;
                added.add(file.getName());
            } else {
                System.out.println(file.getName()+" already indexed, skip");
            }
        }
        node.close();
    }


}
