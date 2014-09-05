package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.ESIndexBuilder;

import java.util.regex.Pattern;

/**
 * Created by chengli on 9/5/14.
 * follow golden_features exp57
 */
public class Exp2 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ESIndex index = loadIndex(config);
        System.out.println(index.getNumDocs());
        index.close();

    }

    static ESIndex loadIndex(Config config) throws Exception{

        ESIndexBuilder builder = ESIndexBuilder.builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setLabelField(config.getString("index.labelField"))
                .setExtLabelField(config.getString("index.extLabelField"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        ESIndex index = builder.build();
        return index;
    }

}
