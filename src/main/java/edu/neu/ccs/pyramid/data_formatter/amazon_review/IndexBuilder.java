package edu.neu.ccs.pyramid.data_formatter.amazon_review;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.List;

/**
 * Created by chengli on 12/21/14.
 */
public class IndexBuilder {
    public static boolean accept(List<String> paragraph){
        boolean cond1 = paragraph.get(6).startsWith("review/score");
        boolean cond2 = paragraph.get(9).startsWith("review/text");
        boolean cond3 = paragraph.get(8).startsWith("review/summary");
        // binary; ignore ambiguous case
        boolean cond4 = Double.parseDouble(paragraph.get(6).split(":")[1])!=3;
        boolean ok = cond1&&cond2&cond3&&cond4;
        return ok;
    }

    static String getFileName(List<String> paragraph) throws Exception{

        return "unknown";
    }

    public static XContentBuilder getBuilder(List<String> paragraph, StanfordCoreNLP nlp, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body", getBody(paragraph,nlp));
        builder.field("file_name",getFileName(paragraph));
        builder.array("label", getLabel(paragraph));
        builder.field("summary",getSummary(paragraph,nlp));
        builder.field("split", getSplit(id));
        builder.field("raw_body", paragraph.get(9));
        builder.field("raw_summary",paragraph.get(8));
        builder.endObject();
        return builder;
    }

    static String getLabel(List<String> paragraph){
        double score = Double.parseDouble(paragraph.get(6).split(":")[1]);
        if (score<3){
            return "neg";
        } else {
            return "pos";
        }
    }


    static String getSplit(int id) {
        String res = null;
        if ( id%5==0){
            res = "test";
        } else {
            res = "train";
        }
        return res;
    }

    static String getBody(List<String> paragraph,StanfordCoreNLP nlp) {
        String text = paragraph.get(9);
        StringBuilder bodyBuilder = new StringBuilder();
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        nlp.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);


        for (CoreMap sentence : sentences) {

            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                bodyBuilder.append(lemma).append(" ");
            }
        }
        return bodyBuilder.toString();
    }

    static String getSummary(List<String> paragraph,StanfordCoreNLP nlp) {
        String text = paragraph.get(8);
        StringBuilder bodyBuilder = new StringBuilder();
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        nlp.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);


        for (CoreMap sentence : sentences) {

            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                bodyBuilder.append(lemma).append(" ");
            }
        }
        return bodyBuilder.toString();
    }


}
