package edu.neu.ccs.pyramid.data_formatter.cnn;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 9/13/14.
 */
public class IndexBuilder {
    static String[] extLabels={"politics","living","health","tech","iBUSINESS",
            "entertainment","iSport","crime","travel"};
    static Map<String,Integer> map = createMap();


    public static XContentBuilder getBuilder(String line) throws Exception{
        String[] split = line.split("\t");
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("file_name",split[0]);
        builder.field("real_label",split[1]);
        builder.field("title",split[2]);
        builder.field("body",split[5]);
        builder.field("label",""+map.get(split[1]));
        builder.field("split",getTrainOrTest());
        builder.endObject();
        return builder;
    }

    static Map<String,Integer> createMap(){
        Map<String,Integer> map = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            map.put(extLabels[i],i);
        }
        return map;
    }

    public static boolean acceptLine(String line){
        String[] split = line.split("\t");
        String realLabel = split[1];
        return (map.containsKey(realLabel)&&(split.length==6)
                &&!split[5].trim().equals(""));
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
}
