package edu.neu.ccs.pyramid.data_formatter.question_classification;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 12/25/14.
 */
public class IndexBuilder {
    static String[] extLabels={"DESC", "ENTY", "ABBR", "HUM","LOC", "NUM"};
    static Map<String,Integer> map = createMap();

    static Map<String,Integer> createMap(){
        Map<String,Integer> map = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            map.put(extLabels[i],i);
        }
        return map;
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    public static XContentBuilder getBuilder(File file, String line, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(line));
        builder.field("file_name",getFileName(file));
        builder.array("real_label", getRealLabel(line));
        builder.array("label",getLabel(line));
        builder.field("split", getSplit(file,id));
        builder.endObject();
        return builder;
    }

    static String getBody(String line){
        int pos = line.indexOf(" ");
        String res = line.substring(pos+1);
        return res;
    }

    static String getLabel(String line){

        String realLabel = getRealLabel(line);
        int label = map.get(realLabel);
        return ""+label;
    }

    static String getRealLabel(String line){
        return line.split(" ")[0].split(":")[0];
    }

    static String getSplit(File file, int id) {
        String res = null;
        if (file.getName().startsWith("train")){
            if ( id%5==0){
                res = "valid";
            } else {
                res = "train";
            }
        } else {
            res = "test";
        }

        return res;
    }
}
