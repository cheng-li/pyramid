package edu.neu.ccs.pyramid.data_formatter.congressional_bill;


import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * Created by Rainicy on 4/23/15.
 */
public class IndexBuilder {


    public static XContentBuilder getBuilder(String docno, String label, String billText, String featureList) throws IOException {

        // parsing billText into text and title string from JSON format
        // parsing featureList into feature field string from JSON format
        ObjectMapper mapper = new ObjectMapper();
        JsonNode actualObj = mapper.readValue(billText, JsonNode.class);
        String title = "";
        String body = "";

        try {
            title = actualObj.get("title").asText();
            body = actualObj.get("text").asText();
        } catch(NullPointerException e) {
            System.out.println("Title or text does not exsit.");
        }

        actualObj = mapper.readValue(featureList, JsonNode.class);
        Iterator<String> iterator = actualObj.fieldNames();
        Set<String> expectedFields = new HashSet<>();
        while (iterator.hasNext()) {
            String field = iterator.next();
            if (acceptField(field)) {
                expectedFields.add(field);
            }
        }

        XContentBuilder builder = getBuilder(docno, label, title, body, expectedFields);

        return builder;


    }

    private static XContentBuilder getBuilder(String docno, String label, String title, String body, Set<String> expectedFields) throws IOException {

        XContentBuilder builder = XContentFactory.jsonBuilder();

        builder.startObject();
        builder.field("docno", docno);
        builder.field("label", label);
        builder.field("title", title);
        builder.field("body", body);

        for (String field : expectedFields) {
            builder.field(field, "1");
        }

        builder.endObject();
        return builder;
    }

    /**
     * if the given field is the expected field.
     * @param field
     * @return
     */
    public static boolean acceptField(String field) {

        if (field.contains("bill_cat4-function")) return false;
        if (field.contains("sim_")) return false;
        if (field.contains("body-word_")) return false;
        if (field.contains("title-word_")) return false;

        return true;
    }

    /**
     * Create the builder for mapping.
     * @return
     * @throws Exception
     */
    public static XContentBuilder createMapping(Set<String> featureFields) throws Exception {
        // mapping the defualt fileds: document number and text.
        XContentBuilder mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject("document")
                .startObject("properties")
                .startObject("docno")
                .field("type", "string")
                .field("store", true)
                .field("index", "not_analyzed")
                .endObject()
                .startObject("title")
                .field("type", "string")
                .field("store", true)
                .field("index", "analyzed")
                .field("term_vector", "with_positions")
                .field("analyzer", "my_english")
                .endObject()
                .startObject("body")
                .field("type", "string")
                .field("store", true)
                .field("index", "analyzed")
                .field("term_vector", "with_positions")
                .field("analyzer", "my_english")
                .endObject();

        for(String feature : featureFields) {
            mapping.startObject(feature)
                    .field("type", "string")
                    .field("store", true)
                    .field("index", "not_analyzed")
                    .field("null_value", "0")
                    .endObject();
        }

        mapping.startObject("label")
                .field("type", "string")
                .field("store", true)
                .field("index", "not_analyzed")
                .endObject();

        mapping.endObject()
                .endObject()
                .endObject();

        return mapping;
    }


    /**
     *  Returns if the given file is a response file.
     * @param file
     * @param response
     * @return if the given file is the response file.
     */
    public static boolean isResponseFile(File file, String response) {
        String fileName = file.getName();
        if(fileName.equals(response)) {
            return true;
        }
        return false;
    }
}
