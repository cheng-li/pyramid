package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.DirWalker;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;

import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import static edu.neu.ccs.pyramid.data_formatter.congressional_bill.IndexBuilder.acceptField;
import static edu.neu.ccs.pyramid.data_formatter.congressional_bill.IndexBuilder.createMapping;
import static edu.neu.ccs.pyramid.data_formatter.congressional_bill.IndexBuilder.getBuilder;
import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * index congressional bill dataset
 * Created by Rainicy on 4/23/15.
 */
public class Exp87 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        // Step 1: getting response/labels and feature list
        System.out.println("Step 1: getting response/labels and feature list.");
        Map<String, String> mapperLabels = new HashMap<>();
        Map<String, String> mapperFeatures = new HashMap<>();
        Set<String> expectedFields = new HashSet<>();

        String responseFile = config.getString("index.responseFile");
        String featureFile = config.getString("index.featureFile");
        String folder = config.getString("index.folder");
        List<File> allFiles = DirWalker.getFiles(folder);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode actualObj;

        for (File f : allFiles) {
            String fName = f.getName();
            // response file
            if (fName.equals(responseFile)) {
                System.out.println("-- dealing with response file: " + f);
                BufferedReader br = new BufferedReader(new FileReader(f));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] docnoLabel = line.split("\\t");
                    mapperLabels.put(docnoLabel[0], docnoLabel[1]);
                }
                br.close();
            } else if (fName.equals(featureFile)) {
                System.out.println("-- dealing with feature file: " + f);
                BufferedReader br = new BufferedReader(new FileReader(f));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] docnoFeature = line.split("\\t");
                    mapperFeatures.put(docnoFeature[0], docnoFeature[1]);

                    // getting expected features
                    actualObj = mapper.readValue(docnoFeature[1], JsonNode.class);
                    Iterator<String> iterator = actualObj.fieldNames();
                    while (iterator.hasNext()) {
                        String field = iterator.next();
                        if (acceptField(field)) {
                            expectedFields.add(field);
                        }
                    }
                }
                br.close();
            }
        }
        System.out.println("number of fields: " + expectedFields.size());
        System.out.println("enter return to continue ......");
        System.in.read();

        // Step 2: Mapping
        System.out.println("Step 2: Mapping ...");
        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        String indexName = config.getString("index.name");
        String indexDocumentType = config.getString("index.documentType");

        // set mapping
        XContentBuilder mappingBuilder = createMapping(expectedFields);
        client.admin().indices().preparePutMapping(indexName)
                .setType(indexDocumentType)
                .setSource(mappingBuilder)
                .execute()
                .actionGet();


        // Step 3: getting the original text file and indexing
        System.out.println("Step 3: getting the original text file and indexing ...");
        int id = 0;

        String textFile = config.getString("index.textFile");
        BufferedReader br = new BufferedReader(new FileReader(textFile));
        String line;
        while ((line = br.readLine()) != null) {
            System.out.println("doc id: " + id);
            String[] docnoText = line.split("\\t");
            String docno = docnoText[0];
            String text = docnoText[1];
            String label = mapperLabels.get(docno);
            String feature = mapperFeatures.get(docno);
            XContentBuilder builder = getBuilder(docno, label, text, feature);

            IndexResponse response = client.prepareIndex(indexName, indexDocumentType, ""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            ++id;
        }
        br.close();

        // close node
        node.close();

    }
}
