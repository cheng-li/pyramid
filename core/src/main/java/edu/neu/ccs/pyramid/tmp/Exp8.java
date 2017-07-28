package edu.neu.ccs.pyramid.tmp;


import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.transport.client.PreBuiltTransportClient;

import java.io.File;
import java.net.InetAddress;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by chengli on 3/4/17.
 */
public class Exp8 {
    public static void main(String[] args) throws Exception{


        String dir  = "/Users/Rainicy/Dropbox/tmp/ohsumed-first-20000-docs/";
        Settings settings = Settings.builder()
                .put("cluster.name", "elasticsearch")
                //.put("client.transport.sniff", true)
                .put("node.name", "ES5")
                //.put("path.home", "/home/dey/es/elasticsearch-5.1.1/data")
                .build();
        //Node node = new Node(settings);
        TransportClient client = new PreBuiltTransportClient(settings)
                .addTransportAddress(new InetSocketTransportAddress(InetAddress.getByName("127.0.0.1"), 9300));

        /*if (true) {
        	SearchResponse response = client.prepareSearch().get();
        	System.out.println(response);
        	System.exit(0);
        }*/
//        Client client = node.client();

        List<File> list = DirWalker.getFiles(dir);
        //make sure each file is indexed only once
        Set<String> added = new HashSet();

        Map<String, Set<String>> nameToCodesMap = IndexBuilder.collectCodes(dir);

        System.out.println(nameToCodesMap); //if (true) System.exit(0);
        int id = 0;
        for (File file: list){
            if (!added.contains(file.getName())){
                // System.out.println("id = "+id);
                XContentBuilder builder = IndexBuilder.getBuilder(file,nameToCodesMap);
                // System.out.println(builder.string());
                IndexResponse response = client.prepareIndex("ohsumed_20000", "document")
                        .setSource(builder)
                        .execute()
                        .actionGet();
                id += 1;
                added.add(file.getName());
            } else {
                System.out.println(file.getName()+" already indexed, skip");
            }
        }
        client.close();
    }
}
