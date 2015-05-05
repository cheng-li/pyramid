package edu.neu.ccs.pyramid.data_formatter.review_polarity;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.apache.commons.io.FileUtils;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

/**
 * Created by chengli on 10/24/14.
 */
public class IndexBuilder {

    static String getBody(File file, StanfordCoreNLP nlp) throws Exception{
        String text = FileUtils.readFileToString(file);
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

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    public static XContentBuilder getBuilder(File file, StanfordCoreNLP nlp, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file,nlp));
        builder.field("file_name",getFileName(file));
        builder.array("label", getRealLabel(file));
        builder.field("split",getTrainOrTest(id));
        builder.endObject();
        return builder;
    }

    static String getLabel(File file){
        String parent = file.getParentFile().getName();
        if (parent.equals("pos")){
            return "1";
        } else {
            return "0";
        }
    }

    static String getRealLabel(File file){
        String parent = file.getParentFile().getName();
        return parent;
    }

    static String getTrainOrTest(int id) {
        String res = null;
        if ( id%5==0){
            res = "test";
        }
        else {
            res = "train";
        }
        return res;
    }
}
