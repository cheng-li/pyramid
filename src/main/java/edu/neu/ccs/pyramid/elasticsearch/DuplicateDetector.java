package edu.neu.ccs.pyramid.elasticsearch;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

/**
 * Created by chengli on 1/20/15.
 */
public class DuplicateDetector implements Serializable{
    private static final long serialVersionUID = 1L;
    private transient ESIndex esIndex;
    Set<String> allDuplicates;

    public DuplicateDetector(ESIndex esIndex) {
        this.esIndex = esIndex;
        this.allDuplicates = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());
    }

    public void addDuplicates(String id1, String id2){
        allDuplicates.add(id1);
        allDuplicates.add(id2);
    }

    public void detect(){
        Map<Integer, Set<String>> lengthToIds = new ConcurrentHashMap<>();
        int numDocs = esIndex.getNumDocs();
        IntStream.range(0,numDocs).parallel()
                .forEach(i -> {
                    Map<Integer,String> termVector = esIndex.getTermVectorFromIndex(""+i);
                    if (!lengthToIds.containsKey(termVector.size())){
                        lengthToIds.put(termVector.size(),Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>()));
                    }
                    lengthToIds.get(termVector.size()).add(""+i);
                });
        lengthToIds.entrySet().stream().parallel().map(Map.Entry::getValue)
                .forEach(this::check);

    }

    private void check(Set<String> candidates){
        List<String> list = new ArrayList<>(candidates);


        int size = candidates.size();
        if (size==1){
            return;
        }
        for (int i=0;i<size-1;i++){
            for (int j=i+i;j<size;j++){
                String id1 = list.get(i);
                String id2 = list.get(j);
                Map<Integer,String> termVector1 = esIndex.getTermVectorFromIndex(id1);
                Map<Integer,String> termVector2 = esIndex.getTermVectorFromIndex(id2);
                if (termVector1.equals(termVector2)){
                    addDuplicates(id1,id2);
                }
            }
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (String str: allDuplicates){
            sb.append(str);
            sb.append(",");
        }
        return sb.toString();
    }
}
