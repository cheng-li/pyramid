package edu.neu.ccs.pyramid.data_formatter.amazon_book_genre;

import org.apache.commons.io.FileUtils;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;

/**
 * Created by chengli on 1/13/15.
 */
public class IndexBuilder {
    static String getBody(File file) throws Exception{
        return FileUtils.readFileToString(file);
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
        switch (parent) {
            case "Biology":
                return "0";
            case "Mathematics":
                return "1";
            default:
                return "2";
        }
    }

    static String getRealLabel(File file){
        String parent = file.getParentFile().getName();
        return parent;
    }

    static String getTrainOrTest(int id) {
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
