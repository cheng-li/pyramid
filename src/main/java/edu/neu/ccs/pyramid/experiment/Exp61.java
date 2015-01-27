package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.io.FileUtils;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * dump most common unigrams
 * Created by chengli on 1/25/15.
 */
public class Exp61 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        SingleLabelIndex index = loadIndex(config);
        build(config,index);
        index.close();
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

    static String[] sampleTrain(Config config, SingleLabelIndex index, Set<String> duplicate){
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        String splitField = config.getString("index.splitField");
        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                .filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("train")).
                        mapToObj(i -> "" + i).filter(id -> !duplicate.contains(id)).collect(Collectors.toList());
        ids = train.toArray(new String[train.size()]);
        return ids;
    }




    static List<String> gatherUnigrams(ESIndex index,
                                       String[] ids, int minDf) throws Exception{
        System.out.println("gathering unigrams with minDf "+minDf);
        Set<TermStat> unigrams = Collections.newSetFromMap(new ConcurrentHashMap<TermStat, Boolean>());
        Arrays.stream(ids).parallel().forEach(id -> {
            Set<TermStat> termStats = null;
            try {
                termStats = index.getTermStats(id);
            } catch (IOException e) {
                e.printStackTrace();
            }
            termStats.stream().filter(termStat -> termStat.getDf() > minDf).forEach(unigrams::add);
        });

        List<String> list = unigrams.stream().sorted(Comparator.comparing(TermStat::getTerm))
                .sorted(Comparator.comparing(TermStat::getDf).reversed())
                .map(TermStat::getTerm)
                .collect(Collectors.toList());
        System.out.println("done");
        System.out.println("there are "+list.size()+" unigrams");
        return list;
    }





    static void build(Config config, SingleLabelIndex index) throws Exception{
        Set<String> duplidate = loadDuplicate(config);
        String[] trainIndexIds = sampleTrain(config,index,duplidate);
        System.out.println("number of training documents = "+trainIndexIds.length);
        List<String> unigrams = gatherUnigrams(index,trainIndexIds,1);
        List<String> frequentUnigrams = unigrams.stream().limit(config.getInt("top")).collect(Collectors.toList());
        File file = new File(config.getString("output.file"));
        file.getParentFile().mkdirs();
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));
        for (String unigram: frequentUnigrams){
            bw.write(unigram);
            bw.write(",");
        }
        bw.close();

    }

    static Set<String> loadDuplicate(Config config) throws Exception{
        File file = new File(config.getString("input.duplicate"));
        String[] strArr = FileUtils.readFileToString(file).split(",");
        Set<String> set = new HashSet<>();
        Arrays.stream(strArr).forEach(set::add);
        return set;
    }
}
