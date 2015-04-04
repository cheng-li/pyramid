package edu.neu.ccs.pyramid.data_formatter.imdb;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.CoreMap;
import org.apache.commons.io.FileUtils;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.Reader;
import java.io.StringReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by chengli on 11/15/14.
 */
public class IndexBuilder {
    static String[] extLabels={"neg","pos"};
    static Map<String,Integer> map = createMap();

    public static boolean acceptFile(File file){
        boolean condition1 = file.getParentFile().getName().equals("pos");
        boolean condition2 = file.getParentFile().getName().equals("neg");
        boolean condition3 = file.getParentFile().getParentFile().getName().equals("train");
        boolean condition4 = file.getParentFile().getParentFile().getName().equals("test");
        boolean accept = (condition1||condition2)&&(condition3||condition4);
        return accept;
    }

    public static XContentBuilder getBuilder(File file, StanfordCoreNLP nlp) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("file_name",getFileName(file));
        builder.field("real_label",getExtLabel(file));
        builder.field("label",""+getLabel(file));
        builder.field("split", getSplit(file));
        setBodyAndPos(file,nlp,builder);
        builder.endObject();
        return builder;
    }





    static void setBodyAndPos(File file, StanfordCoreNLP nlp,XContentBuilder builder) throws Exception{
        String text = FileUtils.readFileToString(file);

        StringBuilder bodyBuilder = new StringBuilder();
        StringBuilder posBuilder = new StringBuilder();
        StringBuilder bodyPosBuilder = new StringBuilder();

        Annotation document = new Annotation(text);

        // run all Annotators on this text
        nlp.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        for(CoreMap sentence: sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token: sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                bodyBuilder.append(lemma).append(" ");

                // this is the POS tag of the token
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                posBuilder.append(pos).append(" ");
                bodyPosBuilder.append(lemma).append(" ").append(pos).append(" ");
            }


        }
        builder.field("raw",text);
        builder.field("body",bodyBuilder.toString());
        builder.field("pos",posBuilder.toString());
        builder.field("body_pos",bodyPosBuilder.toString());
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    static int getLabel(File file) throws Exception{
        String name = file.getParentFile().getName();
        return map.get(name);
    }


    /**
     * fixed
     * @param file
     * @return
     * @throws Exception
     */
    static String getSplit(File file) throws Exception{
        String name = file.getParentFile().getParentFile().getName();
        String res = null;
        switch (name) {
            case "train":
                res = "train";
                break;
            case "test":
                res = "test";
                break;
            default:
                throw new RuntimeException("not train or test");
        }
        return res;
    }

    static String getExtLabel(File file) throws Exception{
        return file.getParentFile().getName();
    }



    static Map<String,Integer> createMap(){
        Map<String,Integer> map = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            map.put(extLabels[i],i);
        }
        return map;
    }
}
