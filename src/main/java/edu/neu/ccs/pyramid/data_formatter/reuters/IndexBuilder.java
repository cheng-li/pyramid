package edu.neu.ccs.pyramid.data_formatter.reuters;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.*;
import java.util.*;

/**
 * Created by Rainicy on 6/26/15.
 */
public class IndexBuilder {

//    public static void main(String[] args) throws ParserConfigurationException, XMLStreamException, SAXException, IOException {
//        String fileName = "/Users/Rainicy/Downloads/5849newsML.xml";
//        File file = new File(fileName);
//        getBuilder(file);
//    }


    public static XContentBuilder getBuilder(File file, Map<String, String> codesDictMap, int id) throws IOException, ParserConfigurationException, SAXException, XMLStreamException {
        XContentBuilder builder = XContentFactory.jsonBuilder();


        ReutersNews reutersNews = getReutersNews(file, codesDictMap);

//        System.out.println(reutersNews);
        builder.startObject();
        builder.field("file_name", reutersNews.fileName);
        builder.field("title", reutersNews.title);
        builder.field("headline", reutersNews.headline);
        builder.field("dateline", reutersNews.dateline);
        builder.field("body", reutersNews.text);
        builder.array("topic_codes", reutersNews.topicCodes);
        builder.field("split", ((id%5) != 0) ? "train" : "test");
        builder.endObject();
        return builder;
    }

    private static ReutersNews getReutersNews(File file, Map<String, String> codesDictMap) throws XMLStreamException, IOException {
        ReutersNews reutersNews = new ReutersNews();
        reutersNews.fileName = file.toString();
        System.out.println(file.toString());
        String tagContent = null;
        String paragraph = null;
        List<String> codes = null;
        Boolean ifText = false;
        Boolean ifTopic = false;
        XMLInputFactory factory = XMLInputFactory.newFactory();
        XMLStreamReader reader =
                factory.createXMLStreamReader(new FileInputStream(file));


        while (reader.hasNext()) {
            int event = reader.next();

            switch (event) {
                case XMLStreamConstants.START_ELEMENT:
                    switch (reader.getLocalName()) {
                        case "text":
                            paragraph = new String();
                            ifText = true;
                            break;
                        case "codes": // only for topic_codes
                            if (reader.getAttributeValue(0).equals("bip:topics:1.0")) {
                                codes = new ArrayList<>();
                                ifTopic = true;
                            }
                            break;
                        case "code":
                            if (ifTopic) {
                                String code = reader.getAttributeValue(0);
                                codes.add(code + " : " + codesDictMap.get(code));
                            }
                            break;
                        case "title":
                            tagContent = "";
                            break;
                        case "headline":
                            tagContent = "";
                            break;
                        case "dateline":
                            tagContent = "";
                            break;
                    }
                    break;


                case XMLStreamConstants.CHARACTERS:
                    if (ifText) {
                        paragraph += reader.getText().trim() + " ";
                    } else {
                        tagContent += reader.getText().trim() + " ";
                    }
                    break;

                case XMLStreamConstants.END_ELEMENT:
                    switch(reader.getLocalName()) {
                        case "title":
                            reutersNews.title = tagContent;
                            break;
                        case "headline":
                            reutersNews.headline = tagContent;
                            break;
                        case "dateline":
                            reutersNews.dateline = tagContent;
                            break;
                        case "text":
                            reutersNews.text = paragraph;
                            ifText = false;
                            break;
                        case "codes":
                            if (ifTopic) {
                                reutersNews.topicCodes =
                                        codes.toArray(new String[codes.size()]);
                                ifTopic = false;
                            }
                            break;
                    }
                    break;
            }
        }
        return reutersNews;
    }

    public static Map<String,String> getCodesDict(String codesDictPath) throws IOException {

        BufferedReader br = new BufferedReader(new FileReader(codesDictPath));

        String line = null;

        Map<String, String> result = new HashMap<>();

        while ((line = br.readLine()) != null) {
            if (line.startsWith(";")) {
                continue;
            }
            try {
                String[] splitInfo = line.split("\\t");
                String code = splitInfo[0];
                String desc = splitInfo[1];
                desc = desc.replace(",", ".");
                result.put(code, desc);
            } catch (Exception e) {
                System.out.println("exception in line: " + line);
                continue;
            }

        }

        br.close();
        return result;
    }

    //
    public static XContentBuilder getBuilder(File file, Map<String, String> codesDictMap, List<String> transLabels, int id) throws IOException, XMLStreamException {
        XContentBuilder builder = XContentFactory.jsonBuilder();

        Set<String> transLabelsSet = new HashSet<>();
        for (String code : transLabels) {
            transLabelsSet.add(code + " : " + codesDictMap.get(code));
        }

        ReutersNews reutersNews = getReutersNews(file, codesDictMap);

//        System.out.println(reutersNews);
        builder.startObject();
        builder.field("file_name", reutersNews.fileName);
        builder.field("title", reutersNews.title);
        builder.field("headline", reutersNews.headline);
        builder.field("dateline", reutersNews.dateline);
        builder.field("body", reutersNews.text);
        builder.array("topic_codes", reutersNews.topicCodes);
        List<String> featureCodes = new ArrayList<>();
        for (int i=0; i<reutersNews.topicCodes.length; i++) {
            String code = reutersNews.topicCodes[i];
            if (transLabelsSet.contains(code)) {
                featureCodes.add(code);
            }
        }
        builder.array("feature_codes", featureCodes.toArray(new String[featureCodes.size()]));
        builder.field("split", ((id%5) != 0) ? "train" : "test");
        builder.endObject();
        return builder;
    }
}

class ReutersNews {
    String fileName;
    String title;
    String headline;
    String dateline;
    String text;
    String[] topicCodes;

    @Override
    public String toString() {
        return "fileName: " + fileName + "\ntitle: " + title + "\nheadline: " + headline +
                "\ndateline: " + dateline + "\ntext: " + text +
                "\ntopicCodes: " + Arrays.toString(topicCodes);
    }
}
