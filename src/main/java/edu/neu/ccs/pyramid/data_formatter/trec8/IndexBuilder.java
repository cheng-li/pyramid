package edu.neu.ccs.pyramid.data_formatter.trec8;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by Rainicy on 4/16/15.
 */
public class IndexBuilder {


    public static List<XContentBuilder> getBuilders(File file, Map<String, HashMap<String, String>> qrelsMap) throws Exception{
        List<XContentBuilder> listBuilders = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = null;

        boolean ifText = false;

        String docno = "";
        String body = "";

        while ((line = br.readLine()) != null) {
            if (line.equals("<DOC>")) {
                docno = "";
                body = "";
            } else if (line.equals("</DOC>")) {
                HashMap<String, String> queryRelevance = new HashMap<>();
                if (qrelsMap.containsKey(docno)) queryRelevance = qrelsMap.get(docno);

                listBuilders.add(getBuilder(docno, body, queryRelevance));
                ifText = false;
            } else if (line.startsWith("<DOCNO>") && line.endsWith("</DOCNO>")) {
                docno = line.replaceAll("<DOCNO>", "");
                docno = docno.replaceAll("</DOCNO>", "");
                docno = docno.trim();
            } else if (line.startsWith("<TEXT>")) {
                ifText = true;
            } else if (line.endsWith("</TEXT>")) {
                ifText = false;
            } else if (line.startsWith("<!--")) {
                continue;
            } else if (line.startsWith("<") && line.endsWith(">")) {
                continue;
            } else {
                if (ifText) {
                    body += " " + line;
                } else {
                    continue;
                }
            }
        }

        br.close();
        return listBuilders;
    }

    public static XContentBuilder getBuilder(String docno, String body, HashMap<String, String> queryRelevance) throws IOException {
        //System.out.println("DOCNO: " + docno);
        //System.in.read();
        //System.out.println("BODY: " + body);
        //System.in.read();
        //System.out.println("RELEVANCE: " + queryRelevance);
        //System.in.read();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("docno", docno);
        builder.field("body", body);
        for (Map.Entry<String, String> pair : queryRelevance.entrySet()) {
            builder.field(pair.getKey(), pair.getValue());
        }
        builder.endObject();
        return builder;
    }

    /**
     * Create the builder for Mapping. Due to the multiple query fileds, we set them as
     * a input Sets to help build mapping.
     *
     * @return XContentBuilder
     * @throws Exception
     */
    public static XContentBuilder creatMapping(Set<String> queryFileds) throws Exception{
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
                            .startObject("body")
                                .field("type", "string")
                                .field("store", true)
                                .field("index", "analyzed")
                                .field("term_vector", "with_positions")
                                .field("analyzer", "my_english")
                            .endObject();

        // mapping other query fields
        for(String query : queryFileds) {
            mapping.startObject(query)
                    .field("type", "string")
                    .field("store", true)
                    .field("index", "not_analyzed")
                   .endObject();
        }

        mapping.endObject().endObject().endObject();

        return mapping;
    }

    /**
     * Set up a series of rules to check if the given file is
     * acceptable.
     * @param file
     * @return
     */
    public static boolean acceptFile(File file) {
        String fileName = file.getName();
        if (fileName.contains(".h")) return false;
        if (fileName.contains(".c")) return false;
        if (fileName.contains("makefile")) return false;
        if (fileName.contains("read")) return false;

        return true;
    }
}
