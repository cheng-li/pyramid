package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.wipo.IndexBuilder;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index wipo
 * Created by chengli on 12/27/14.
 */
public class Exp45 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File trainFile = new File(config.getString("input.trainFile"));
        File testFile = new File(config.getString("input.testFile"));
        IndexBuilder indexBuilder = new IndexBuilder(trainFile,testFile,config.getString("section").toUpperCase());

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;

        try(BufferedReader br = new BufferedReader(new FileReader(trainFile))
        ){
            String line;
            while((line=br.readLine())!=null){
                if (indexBuilder.accept(line)){
                    System.out.println("id="+id);
                    XContentBuilder builder = indexBuilder.getBuilder(trainFile, line, id);
                    IndexResponse response = client.prepareIndex("wipo-"+config.getString("section").toLowerCase(), "document",""+id)
                            .setSource(builder)
                            .execute()
                            .actionGet();
                    id += 1;
                }

            }
        }

        try(BufferedReader br = new BufferedReader(new FileReader(testFile))
        ){
            String line;
            while((line=br.readLine())!=null){
                if (indexBuilder.accept(line)){
                    System.out.println("id="+id);
                    XContentBuilder builder = indexBuilder.getBuilder(testFile, line, id);
                    IndexResponse response = client.prepareIndex("wipo-"+config.getString("section").toLowerCase(), "document",""+id)
                            .setSource(builder)
                            .execute()
                            .actionGet();
                    id += 1;
                }

            }
        }

        node.close();
    }
}
