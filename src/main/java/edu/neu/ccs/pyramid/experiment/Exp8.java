package edu.neu.ccs.pyramid.experiment;

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
import java.util.List;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index Ohsumed_20000 on fiji
 * Created by chengli on 10/1/14.
 */
public class Exp8 {
    public static void main(String[] args) throws Exception{
        String dir  = "/huge1/people/chengli/Datasets/Ohsumed/original/ohsumed-first-20000-docs";

        Node node = nodeBuilder().client(true).clusterName("fijielasticsearch").node();
        Client client = node.client();

        List<File> list = DirWalker.getFiles(dir);
        int id = 0;
        for (File file: list){
            System.out.println("id = "+id);
            XContentBuilder builder = IndexBuilder.getBuilder(file);
//            System.out.println(builder.string());
            IndexResponse response = client.prepareIndex("ohsumed_20000", "document",""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            id += 1;
        }
        node.close();
    }


}
