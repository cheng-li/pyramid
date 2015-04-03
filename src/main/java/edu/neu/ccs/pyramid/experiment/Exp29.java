package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.imdb.IndexBuilder;
import edu.neu.ccs.pyramid.data_formatter.imdb.SentencesIndexer;
import edu.neu.ccs.pyramid.util.DirWalker;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index imdb_stopwords
 * Created by chengli on 11/22/14.
 */
public class Exp29 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        String folder = new File(config.getString("input.documentFolder")).getAbsolutePath();
        String taggerModel = config.getString("input.taggerModel");

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        props.setProperty("pos.model",taggerModel);
        StanfordCoreNLP nlp = new StanfordCoreNLP(props);

//        String sentencesFolder = new File(config.getString("input.sentenceFolder")).getAbsolutePath();
        List<File> files = DirWalker.getFiles(folder);


        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        int id = 0;
        for (File file: files){
            if (SentencesIndexer.acceptFile(file)){
                System.out.println("id = "+id);

//                String sentenceFileName = file.getAbsolutePath().replaceFirst(folder,sentencesFolder);
                XContentBuilder builder = IndexBuilder.getBuilder(file, nlp);
                //               System.out.println(builder.string());
                IndexResponse response = client.prepareIndex("imdb", "document",""+id)
                        .setSource(builder)
                        .execute()
                        .actionGet();
                id += 1;
            }

        }

        node.close();
    }
}
