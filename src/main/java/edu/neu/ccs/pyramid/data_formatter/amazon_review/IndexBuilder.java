package edu.neu.ccs.pyramid.data_formatter.amazon_review;

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
        // binary; ignore ambiguous case
        boolean cond3 = Double.parseDouble(paragraph.get(6).split(":")[1])!=3;
        boolean ok = cond1&&cond2&cond3;
        return ok;
    }

    static String getFileName(List<String> paragraph) throws Exception{

        return "unknown";
    }

    public static XContentBuilder getBuilder(List<String> paragraph, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",paragraph.get(9));
        builder.field("file_name",getFileName(paragraph));
        builder.array("real_label", getRealLabel(paragraph));
        builder.array("label",getLabel(paragraph));
        builder.field("split", getSplit(id));
        builder.endObject();
        return builder;
    }

    static String getLabel(List<String> paragraph){
        double score = Double.parseDouble(paragraph.get(6).split(":")[1]);
        if (score<3){
            return "0";
        } else {
            return "1";
        }
    }

    static String getRealLabel(List<String> paragraph){
        double score = Double.parseDouble(paragraph.get(6).split(":")[1]);
        if (score<3){
            return "negative";
        } else {
            return "positive";
        }
    }

    static String getSplit(int id) {
        String res = null;
        if ( id%5==0){
            res = "test";
        } else if (id%5==1){
            res = "valid";
        } else {
            res = "train";
        }
        return res;
    }


}
