package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * write data to the seql format
 * Created by chengli on 1/29/15.
 */
public class Exp64 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        SingleLabelIndex index = loadIndex(config);

        dumpTrain(config,index);
        dumpTest(config,index);

    }


    static SingleLabelIndex loadIndex(Config config) throws Exception{
        SingleLabelIndex.Builder builder = new SingleLabelIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setLabelField(config.getString("index.labelField"))
                .setExtLabelField(config.getString("index.extLabelField"))
                .setDocumentType(config.getString("index.documentType"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        SingleLabelIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
    }

    static void dumpTrain(Config config, SingleLabelIndex index) throws Exception{
        int numDocsInIndex = index.getNumDocs();
        String splitField = config.getString("index.splitField");
        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                .filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("train")).
                        mapToObj(i -> "" + i).collect(Collectors.toList());

        File folder = new File(config.getString("output.folder"));
        folder.mkdirs();
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(config.getString("output.folder"),"train")));
        for (String id: train){
            int label = index.getLabel(id);
            if (label==0){
                bw.write("-1");
            } else {
                bw.write("+1");
            }

            bw.write(" ");
            String body = index.getStringField(id,index.getBodyField()).replaceAll("\n"," ");
            bw.write(body);
            bw.write("\n");
        }
        bw.close();
    }

    static void dumpTest(Config config, SingleLabelIndex index) throws Exception{
        File folder = new File(config.getString("output.folder"));
        folder.mkdirs();
        int numDocsInIndex = index.getNumDocs();
        String splitField = config.getString("index.splitField");
        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                .filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("test")).
                        mapToObj(i -> "" + i).collect(Collectors.toList());

        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(config.getString("output.folder"),"test")));
        for (String id: train){
            int label = index.getLabel(id);
            if (label==0){
                bw.write("-1");
            } else {
                bw.write("+1");
            }
            bw.write(" ");
            String body = index.getStringField(id,index.getBodyField()).replaceAll("\n"," ");
            bw.write(body);
            bw.write("\n");
        }
        bw.close();
    }
}
