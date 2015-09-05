package edu.neu.ccs.pyramid.experiment;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * get frequent nouns
 * Created by chengli on 4/11/15.
 */
public class Exp80 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        SingleLabelIndex index = loadIndex(config);
        Multiset<String> frequentNouns = frequentNoun(index);
        Comparator<Multiset.Entry<String>> comparator = Comparator.comparing(Multiset.Entry::getCount);
        List<Multiset.Entry<String>> list =  frequentNouns.entrySet().stream()
                .sorted(comparator.reversed()).collect(Collectors.toList());
        System.out.println(list);
        index.close();

    }


    private static Multiset<String> frequentNoun(ESIndex index){
        Multiset<String> multiset = ConcurrentHashMultiset.create();
        IntStream.range(0,index.getNumDocs()).parallel().forEach( i -> {
            String docid = ""+i;
            Map<Integer,String> posTermVector = index.getTermVectorFromIndex("pos",docid);
            Map<Integer,String> bodyTermVector = index.getTermVectorFromIndex("body",docid);
            parseOneDoc(multiset,posTermVector,bodyTermVector);
        });

        return multiset;
    }

    private static void parseOneDoc(Multiset<String> multiset, Map<Integer,String> posTermVector, Map<Integer,String> bodyTermVector){
        for (Map.Entry<Integer,String> entry: posTermVector.entrySet()){
            int position = entry.getKey();
            String tag = entry.getValue();
            boolean cond1 = tag.equals("NN");
            boolean cond2 = tag.equals("NNS");
            boolean cond3 = tag.equals("NNS");
            boolean cond4 = tag.equals("NNP");
            boolean cond5 = tag.equals("NNPS");
            boolean cond = cond1 || cond2 || cond3 || cond4 || cond5;
            if (cond){
                multiset.add(bodyTermVector.get(position));
            }
        }
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
}
