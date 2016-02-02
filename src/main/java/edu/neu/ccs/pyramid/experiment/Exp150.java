package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.data_formatter.reuters21578.IndexBuilder;
import edu.neu.ccs.pyramid.util.DirWalker;
import org.apache.commons.io.FileUtils;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.node.Node;

import java.io.File;
import java.util.*;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by chengli on 2/1/16.
 */
public class Exp150 {
    public static void main(String[] args) throws Exception{
        Map<String,String[]> map = loadLabelMap();
        Node node = nodeBuilder().client(true).clusterName("elasticsearch").node();

        Client client = node.client();
        String dir1 = "/Users/chengli/Dropbox/Shared/pyramid_shared/Datasets/reuters21578/training";
        String dir2 = "/Users/chengli/Dropbox/Shared/pyramid_shared/Datasets/reuters21578/test";
        List<File> list = DirWalker.getFiles(dir1);
        list.addAll(DirWalker.getFiles(dir2));

        int id = 0;
        for (File file: list){
            System.out.println("id = "+id);
            XContentBuilder builder = IndexBuilder.getBuilder(file,map);
//               System.out.println(builder.string());
            IndexResponse response = client.prepareIndex("reuters21578", "document")
                    .setSource(builder)
                    .execute()
                    .actionGet();
                id += 1;
        }
        node.close();
    }

    private static Map<String,String[]> loadLabelMap() throws Exception{
        String file = "/Users/chengli/Dropbox/Shared/pyramid_shared/Datasets/reuters21578/cats.txt";
        Map<String,String[]> map = new HashMap<>();
        List<String> lines = FileUtils.readLines(new File(file));
        for (String line: lines){
            String[] split = line.split(" ");
            int numLables = split.length-1;
            String[] labels = new String[numLables];
            for (int i=1;i<split.length;i++){
                labels[i-1]=split[i];
            }
            map.put(split[0], labels);
            System.out.println(split[0]);
            System.out.println(Arrays.toString(labels));
        }
        return map;
    }
}
