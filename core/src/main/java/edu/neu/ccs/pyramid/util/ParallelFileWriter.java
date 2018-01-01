package edu.neu.ccs.pyramid.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParallelFileWriter {
    public static <T> void mapToString(ParallelStringMapper writable, List<T> list, File file, int batchSize) throws IOException {
        int l = (int)Math.ceil((double)list.size()/batchSize);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))){
            for (int i=0; i<l; i++){
                List<String> stringList = IntStream.range(i*batchSize, Math.min((i+1)*batchSize, list.size())).parallel()
                        .mapToObj(j->writable.mapToString(list,j)).collect(Collectors.toList());
                for(String str:stringList){
                    bw.write(str);
                }
            }

        }

    }
}
