package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.BufferedReader;
import java.io.FileReader;

import static edu.neu.ccs.pyramid.data_formatter.yelp.IndexBuilder.createMapping;
import static edu.neu.ccs.pyramid.data_formatter.yelp.IndexBuilder.createBuilder;
import static org.elasticsearch.node.NodeBuilder.nodeBuilder;


/**
 * index whole yelp
 * Created by Rainicy on 5/1/15.
 */
public class Exp95 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        System.out.println("Step 1: Mapping ...");

        Node node = nodeBuilder().client(true).clusterName(config.getString("index.clusterName")).node();
        Client client = node.client();
        String indexName = config.getString("index.name");
        String indexDocumentType = config.getString("index.documentType");

        // set mapping
        XContentBuilder mappingBuilder = createMapping();

        client.admin().indices().preparePutMapping(indexName)
                .setType(indexDocumentType)
                .setSource(mappingBuilder)
                .execute()
                .actionGet();

        System.out.println("Step 2: Index ...");
        String reviewFile = config.getString("index.reviewFile");
        BufferedReader br = new BufferedReader(new FileReader(reviewFile));
        String line;
        ObjectMapper mapper = new ObjectMapper();
        JsonNode actualObj = null;
        int id = 0;

        while((line = br.readLine()) != null) {
            actualObj = mapper.readValue(line, JsonNode.class);

            int stars = actualObj.get("stars").asInt();
            String label = "";
            if (stars < 3) {
                label = "0";
            } else if (stars > 3) {
                label = "1";
            } else {
                continue;
            }

            String reviewId = actualObj.get("review_id").asText();
            String body = actualObj.get("text").asText();

            System.out.println("id : " + id);

            XContentBuilder builder = createBuilder(id, reviewId, body, label);
            IndexResponse response = client.prepareIndex(indexName, indexDocumentType, ""+id)
                    .setSource(builder)
                    .execute()
                    .actionGet();
            ++id;

        }
        br.close();
        node.close();


    }
}
