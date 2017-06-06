package edu.neu.ccs.pyramid.core.application;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.elasticsearch.ESIndex;

import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by chengli on 2/10/15.
 */
public class IndexChecker {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        ESIndex index = loadIndex(config);
        List<String> fields = config.getStrings("fieldsToCheck");
        for (String field: fields){
            check(index,field);
        }
        for (String field: fields){
            checkEmpty(index,field);
        }
        index.close();


    }

    private static void check(ESIndex index, String field){
        List<String> ids = index.getAllDocs().stream()
                .filter(id -> index.getField(id, field) == null)
                .collect(Collectors.toList());
        if (ids.size()==0){
            System.out.println("all documents have the field "+field);
        } else {
            System.out.println("the following documents miss the field "+field);
            System.out.println(ids);
        }


    }

    private static void checkEmpty(ESIndex index, String field){
        List<String> ids = index.getAllDocs().stream()
                .filter(id -> index.getField(id, field) != null && ((String) index.getField(id, field)).trim().equals(""))
                .collect(Collectors.toList());
        if (ids.size()==0){
            System.out.println("all documents have non-empty field "+field);
        } else {
            System.out.println("the following documents have empty field "+field);
            System.out.println(ids);
        }


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
