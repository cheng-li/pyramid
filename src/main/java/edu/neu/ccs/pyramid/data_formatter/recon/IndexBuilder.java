package edu.neu.ccs.pyramid.data_formatter.recon;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.Map;
import java.util.Set;

/**
 * Created by chengli on 4/27/15.
 */
public class IndexBuilder {
    public static XContentBuilder getBuilder(String line, String label) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",line);
        builder.array("label", label);
        builder.endObject();
        return builder;
    }
}
