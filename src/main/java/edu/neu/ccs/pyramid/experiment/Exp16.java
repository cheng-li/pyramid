package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;

import edu.neu.ccs.pyramid.data_formatter.review_polarity.IndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.File;
import java.util.List;
import java.util.Properties;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index review polarity
 * Created by chengli on 10/24/14.
 */
public class Exp16 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String dir = config.getString("input.folder");
        String taggerModel = config.getString("input.taggerModel");

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        props.setProperty("pos.model",taggerModel);
        StanfordCoreNLP nlp = new StanfordCoreNLP(props);

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();

        List<File> list = DirWalker.getFiles(dir);
        int id = 0;
        for (File file : list) {
            System.out.println("id = "+id);
            XContentBuilder builder = IndexBuilder.getBuilder(file, nlp,id);
            //               System.out.println(builder.string());
            IndexResponse response = client.prepareIndex("review_polarity", "document",""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            id += 1;
        }

        node.close();
    }
}
