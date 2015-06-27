package edu.neu.ccs.pyramid.data_formatter.reuters;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Rainicy on 6/26/15.
 */
public class IndexBuilder {

//    public static void main(String[] args) throws ParserConfigurationException, XMLStreamException, SAXException, IOException {
//        String fileName = "/Users/Rainicy/Downloads/287192newsML.xml";
//        File file = new File(fileName);
//        getBuilder(file);
//    }

    public static XContentBuilder getBuilder (File file) throws IOException, ParserConfigurationException, SAXException, XMLStreamException {
        XContentBuilder builder = XContentFactory.jsonBuilder();


        ReutersNews reutersNews = getReutersNews(file);

//        System.out.println(reutersNews);


        builder.startObject();
        builder.field("fileName", reutersNews.fileName);
        builder.field("title", reutersNews.title);
        builder.field("headline", reutersNews.headline);
        builder.field("dateline", reutersNews.dateline);
        builder.field("body", reutersNews.text);
        builder.array("topic_codes", reutersNews.topicCodes);
        builder.field("split", (Math.random() < 0.8) ? "train" : "test");
        builder.endObject();
        return builder;
    }

    private static ReutersNews getReutersNews(File file) throws XMLStreamException, FileNotFoundException {
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
                                codes.add(reader.getAttributeValue(0));
                            }
                            break;
                    }
                    break;


                case XMLStreamConstants.CHARACTERS:
                    if (ifText) {
                        paragraph += reader.getText().trim() + " ";
                    } else {
                        tagContent = reader.getText().trim();
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
