package edu.neu.ccs.pyramid.data_formatter.blog_gender;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 12/26/14.
 */
public class IndexBuilder {
    static String[] extLabels={"F", "M"};
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
        builder.field("split", getSplit(id));
        builder.endObject();
        return builder;
    }

    static String getBody(String line){
        int pos = line.indexOf("\t");
        String res = line.substring(pos+1);
        return res;
    }

    static String getLabel(String line){

        String realLabel = getRealLabel(line);
        int label = map.get(realLabel);
        return ""+label;
    }

    static String getRealLabel(String line){
        return line.split("\t")[0].trim().toUpperCase();
    }

    static String getSplit(int id) {
        String res = null;
        if ( id%5==0){
            res = "valid";
        } else if ( id%5==1){
            res = "train";
        } else {
        res = "test";
        }

        return res;
    }
}
