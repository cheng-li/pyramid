package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;

import static edu.neu.ccs.pyramid.data_formatter.trec8.IndexBuilder.acceptFile;
import static edu.neu.ccs.pyramid.data_formatter.trec8.IndexBuilder.creatMapping;
import static edu.neu.ccs.pyramid.data_formatter.trec8.IndexBuilder.getBuilders;
import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Build trec8 index.
 * Created by Rainicy on 4/16/15.
 */
public class Exp86 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        // reads the qrels file
        String qrelsFile = config.getString("index.qrelsFile");
        Map<String, HashMap<String, String>> qrelsMap = new HashMap<>();
        Set<String> querySet = new HashSet<>();
        BufferedReader br = new BufferedReader(new FileReader(qrelsFile));
        String line = null;
        while ((line = br.readLine()) != null ) {
            //System.out.println(line);
            //System.in.read();
            String[] info = line.split(" "); //query_id foo docno relevance
            String docno = info[2];
            String queryId = info[0];
            String relevance = info[3];

            //System.out.println("DOCNO: " + docno + "; QueryID: " + queryId + "; Rel: " + relevance);
            //System.in.read();
            // update query set
            if (!querySet.contains(queryId)) {
                querySet.add(queryId);
            }


            if(qrelsMap.containsKey(docno)) {
                HashMap<String, String> temp = qrelsMap.get(docno);
                temp.put(queryId, relevance);
                qrelsMap.put(docno, temp);
            } else {
                HashMap<String, String> temp = new HashMap<>();
                temp.put(queryId, relevance);
                qrelsMap.put(docno, temp);
            }
        }
        br.close();

        //System.out.println("QUERY ID:");
        //System.out.println(querySet);
        //System.in.read();
        //System.out.println("QRELS MAP");
        //System.out.println(qrelsMap);
        //System.in.read();

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        String indexName = config.getString("index.name");
        String indexDocumentType = config.getString("index.documentType");

        // Set mapping
        XContentBuilder mappingBuilder = creatMapping(querySet);
        client.admin().indices().preparePutMapping(indexName)
                .setType(indexDocumentType)
                .setSource(mappingBuilder)
                .execute()
                .actionGet();

        int id = 0;

        List<String> folders = config.getStrings("index.dataFolder");
        for (String folder : folders) {
            List<File> files = DirWalker.getFiles(folder);
            for (File file : files) {
                if(acceptFile(file)) {
                    System.out.println("File name:" + file);
                    //System.in.read();

                    List<XContentBuilder> builders = getBuilders(file, qrelsMap);
                    for (XContentBuilder builder : builders) {
                        System.out.println("ID: " + id);
                        IndexResponse response = client.prepareIndex(indexName, indexDocumentType, "" + id)
                                .setSource(builder)
                                .execute()
                                .actionGet();
                        ++id;
                    }
                }
            }
        }

        // shutdown
        node.close();
    }
}
