package edu.neu.ccs.pyramid.util;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ParallelFileWriterTest {
    public static void main(String[] args) throws IOException {
        List<Integer> list = IntStream.range(0,1007).boxed().collect(Collectors.toList());
        File file = new File("/Users/zhenmingbi/tmp/aa.txt");
        ParallelStringMapper<Integer> writable = (list1, i) -> {
            if(list1.get(i)%2==0){
                return "The "+ i +"th element is even\n";
            }else{
                return "The "+ i +"th element is odd\n";
            }
        };
        ParallelFileWriter.mapToString(writable, list, file, 100);
    }

}