package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index amazon_phone, class demo
 * Created by chengli on 5/11/15.
 */
public class Exp103 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        String fileName = config.getString("input.file");
        String indexName = config.getString("index.name");

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();


        int id = 0;
        File file = new File(fileName);
        List<String> document = new ArrayList<>();

        try(BufferedReader br = new BufferedReader(new FileReader(file))
        ){
            String line;
            while((line=br.readLine())!=null){
                if (!line.equals("")){
                    document.add(getValue(line));
                } else {
                    System.out.println("id = "+ id);
                    XContentBuilder builder = getBuilder(document);
                    IndexResponse response = client.prepareIndex(indexName, "document",""+id)
                            .setSource(builder)
                            .execute()
                            .actionGet();
                    id += 1;
                    document = new ArrayList<>();
                }
            }
        }


        node.close();
    }





    public static XContentBuilder getBuilder(List<String> document) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("productId", document.get(0));
        builder.field("title", document.get(1));
        builder.field("price",document.get(2));
        builder.field("userId", document.get(3));
        builder.field("profileName",document.get(4));
        builder.field("helpfulness_numerator", document.get(5).split("/")[0]);
        builder.field("helpfulness_denominator",document.get(5).split("/")[1]);
        builder.field("score", (int)Double.parseDouble(document.get(6)));
        builder.field("time", document.get(7));
        builder.field("summary",document.get(8));
        builder.field("text",document.get(9));
        builder.endObject();
        return builder;
    }

    public static String getValue(String line){
        int start = line.indexOf(":");
        return line.substring(start+2);
    }

}
