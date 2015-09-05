package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.amazon_review.IndexBuilder;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index amazon review
 * Created by chengli on 12/21/14.
 */
public class Exp40 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        String fileName = config.getString("input.file");
        String indexName = config.getString("index.name");


        String taggerModel = config.getString("input.taggerModel");

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        props.setProperty("pos.model",taggerModel);
        StanfordCoreNLP nlp = new StanfordCoreNLP(props);


        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;
        File file = new File(fileName);
        List<String> paragragh = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader(file))
        ){
            String line;
            while((line=br.readLine())!=null){
                if (!line.equals("")){
                    paragragh.add(line);
                } else {
                    if (IndexBuilder.accept(paragragh)){
                        System.out.println("id = "+ id);
                        XContentBuilder builder = IndexBuilder.getBuilder(paragragh, nlp,id);
                        IndexResponse response = client.prepareIndex(indexName, "document",""+id)
                                .setSource(builder)
                                .execute()
                                .actionGet();
                        id += 1;
                    } else {
                        System.out.println("reject");
//                        System.out.println(paragragh.toString());
                    }
                    paragragh = new ArrayList<>();
                }
            }
        }


        node.close();
    }

}
