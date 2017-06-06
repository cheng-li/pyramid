package edu.neu.ccs.pyramid.dataset;

import java.io.IOException;

/**
 * Created by Rainicy on 10/31/15.
 */
public class MekaFormatTest {
    public static void main(String[] args) throws IOException {
        MultiLabelClfDataSet dataSet = MekaFormat.loadMLClfDataset("/Users/Rainicy/Downloads/corel5k-sparse/Corel5k-sparse.arff", 499, 374, "sparse");
        TRECFormat.save(dataSet, "/Users/Rainicy/Downloads/corel5k-sparse/sparse_trec");
    }
}

