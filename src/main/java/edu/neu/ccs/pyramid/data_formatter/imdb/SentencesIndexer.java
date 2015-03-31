package edu.neu.ccs.pyramid.data_formatter.imdb;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import sun.nio.ch.IOUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by chengli on 2/12/15.
 */
public class SentencesIndexer {
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

    public static XContentBuilder getBuilder(File file, File sentenceFile) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.field("real_label",getExtLabel(file));
        builder.field("label",""+getLabel(file));
        builder.field("split", getSplit(file));
        builder.field("first_sentence",getFirstSentence(sentenceFile));
        builder.field("last_sentence",getLastSentence(sentenceFile));
        builder.endObject();
        return builder;
    }



    static String getBody(File file) throws Exception{
        String entireFileText = FileUtils.readFileToString(file)
                .replaceAll("\\?", " punctuationquestion ")
                .replaceAll("!"," punctuationexclamation ");;
        return entireFileText;
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    static int getLabel(File file) throws Exception{
        String name = file.getParentFile().getName();
        return map.get(name);
    }

    static String getFirstSentence(File sentenceFile) throws Exception{
        BufferedReader br = new BufferedReader(new FileReader(sentenceFile));
        List<String> lines = IOUtils.readLines(br);
        br.close();
        return lines.get(0);
    }

    static String getLastSentence(File sentenceFile) throws Exception{
        BufferedReader br = new BufferedReader(new FileReader(sentenceFile));
        List<String> lines = IOUtils.readLines(br);
        br.close();
        return lines.get(lines.size()-1);
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
