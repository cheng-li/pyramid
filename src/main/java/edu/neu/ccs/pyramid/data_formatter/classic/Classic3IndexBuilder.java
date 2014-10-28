package edu.neu.ccs.pyramid.data_formatter.classic;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Pattern;

/**
 * ignore CACM
 * Created by chengli on 10/28/14.
 */
public class Classic3IndexBuilder {
    static String[] extLabels={"cisi","cran","med"};
    static Map<String,Integer> map = createMap();

    static String getBody(File file) throws Exception{
        String entireFileText = new Scanner(file)
                .useDelimiter("\\A").next();
        return entireFileText;
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    public static XContentBuilder getBuilder(File file) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.array("real_label", getRealLabel(file));
        builder.array("label",getLabel(file));
        builder.field("split",getTrainOrTest());
        builder.endObject();
        return builder;
    }

    static String getLabel(File file){
        String start = file.getName().split(Pattern.quote("."))[0];
        return map.get(start).toString();
    }

    static String getRealLabel(File file){
        return file.getName().split(Pattern.quote("."))[0];
    }

    static String getTrainOrTest() {
        String res = null;
        double ran = Math.random();

        if ( ran< 0.6){
            res = "train";
        } else {
            res = "test";
        }
        return res;
    }

    static Map<String,Integer> createMap(){
        Map<String,Integer> map = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            map.put(extLabels[i],i);
        }
        return map;
    }

}
