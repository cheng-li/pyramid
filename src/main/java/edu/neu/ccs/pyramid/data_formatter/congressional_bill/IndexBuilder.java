package edu.neu.ccs.pyramid.data_formatter.congressional_bill;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;

/**
 * Created by Rainicy on 4/23/15.
 */
public class IndexBuilder {





    /**
     * Create the builder for mapping.
     * @return
     * @throws Exception
     */
    public static XContentBuilder createMapping() throws Exception {
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
                .endObject()
                .endObject()
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
