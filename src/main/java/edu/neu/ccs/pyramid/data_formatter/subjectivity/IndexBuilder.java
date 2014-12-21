package edu.neu.ccs.pyramid.data_formatter.subjectivity;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;

/**
 * Created by chengli on 12/21/14.
 */
public class IndexBuilder {
    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    public static XContentBuilder getBuilder(File file, String line, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",line);
        builder.field("file_name",getFileName(file));
        builder.array("real_label", getRealLabel(file));
        builder.array("label",getLabel(file));
        builder.field("split", getSplit(id));
        builder.endObject();
        return builder;
    }

    static String getLabel(File file){
        if (file.getName().startsWith("plot")){
            return "1";
        } else {
            return "0";
        }
    }

    static String getRealLabel(File file){
        if (file.getName().startsWith("plot")){
            return "objective";
        } else {
            return "subjective";
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
