package edu.neu.ccs.pyramid.data_formatter.review_polarity;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

/**
 * Created by chengli on 10/24/14.
 */
public class IndexBuilder {

    static String getBody(File file) throws Exception{
        String entireFileText = new Scanner(file)
                .useDelimiter("\\A").next();
        return entireFileText;
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    public static XContentBuilder getBuilder(File file, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.array("real_label", getRealLabel(file));
        builder.array("label",getLabel(file));
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
        if ( id%10==0){
            res = "test";
        } else {
            res = "train";
        }
        return res;
    }
}
