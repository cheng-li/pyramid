package edu.neu.ccs.pyramid.experiment;


import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.node.Node;

import java.io.File;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index imdb toy
 * Created by chengli on 7/21/15.
 */
public class Exp131 {
    public static void main(String[] args) throws Exception{
        String[] docs = {"I highly recommend this movie",
                "I do not recommend this movie to anybody",
                "It is a waste of time",
                "Good fun stuff",
                "It's just not worth your time",
                "I do not recommend this movie unless you are prepared for the biggest waste of money and time of your life",
                "This movie was the slowest and most boring so called horror that I have ever seen",
                "All i can say is that is movie was horrible",
                "A wonderful story and a wonderful film",
                "This is a really nice and sweet movie that the entire family can enjoy"};
        String[] labels = {"pos","neg","neg","pos","neg",  "neg","neg","neg","pos","pos"};


        Node node = nodeBuilder().client(true).clusterName("fijielasticsearch").node();
        Client client = node.client();
        for (int id=0;id<10;id++){
            System.out.println("id = "+id);

            XContentBuilder builder = XContentFactory.jsonBuilder();
            builder.startObject();
            builder.field("label",labels[id]);
            String split;
            if (id<5){
                split="train";
            } else {
                split = "test";
            }
            builder.field("split", split);
            builder.field("body",docs[id]);
            builder.endObject();
            //               System.out.println(builder.string());
            IndexResponse response = client.prepareIndex("imdb_toy", "document",""+(id+1))
                    .setSource(builder)
                    .execute()
                    .actionGet();

        }
        node.close();
    }

}
