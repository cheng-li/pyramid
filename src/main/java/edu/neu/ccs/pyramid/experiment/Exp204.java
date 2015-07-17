package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.MultiLabelIndex;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.regex.Pattern;

/**
 * Created by Rainicy on 7/15/15.
 */
public class Exp204 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the conifg file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ESIndex index = loadIndex(config);

        String outputFolder = config.getString("output.folder");
        BufferedWriter bwTrainP = new BufferedWriter(new FileWriter(outputFolder + "training_paragraphs.txt"));
        BufferedWriter bwTrainL = new BufferedWriter(new FileWriter(outputFolder + "training_labels.txt"));
        BufferedWriter bwTestP = new BufferedWriter(new FileWriter(outputFolder + "testing_paragraphs.txt"));
        BufferedWriter bwTestL = new BufferedWriter(new FileWriter(outputFolder + "testing_labels.txt"));

        for (int i=0; i<index.getNumDocs(); i++) {
            System.out.println("Working on doc id :\t" + i);
            String body = index.getStringField(String.valueOf(i), "body").replaceAll("\\n", " ");
            String label = index.getStringField(String.valueOf(i), "label");
            String split = index.getStringField(String.valueOf(i), "split").replaceAll("\\n", "");

            if (split.equals("train")) {
                bwTrainP.write(body + "\n");
                bwTrainL.write(label + "\n");
            } else {
                bwTestP.write(body + "\n");
                bwTestL.write(label + "\n");
            }
        }

        bwTrainP.close();
        bwTrainL.close();
        bwTestP.close();
        bwTestL.close();
    }

    static ESIndex loadIndex(Config config) throws Exception{
        ESIndex.Builder builder = new ESIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setDocumentType(config.getString("index.documentType"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        ESIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
        return index;
    }
}

