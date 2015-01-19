package edu.neu.ccs.pyramid.data_formatter.imdb;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.HashMap;
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

    public static XContentBuilder getBuilder(File file) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.field("real_label",getExtLabel(file));
        builder.field("label",""+getLabel(file));
        builder.field("split", getSplit(file));
        builder.endObject();
        return builder;
    }



    static String getBody(File file) throws Exception{
        String entireFileText = new Scanner(file)
                .useDelimiter("\\A").next();
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
