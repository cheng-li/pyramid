package edu.neu.ccs.pyramid.core.elasticsearch;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 1/20/15.
 */
public class DuplicateDetector implements Serializable{
    private static final long serialVersionUID = 1L;
    private transient ESIndex esIndex;
    Set<String> allDuplicates;
    private String splitField;

    public Set<String> getAllDuplicates() {
        return allDuplicates;
    }

    public DuplicateDetector(ESIndex esIndex, String splitField) {
        this.esIndex = esIndex;
        this.allDuplicates = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());
        this.splitField = splitField;
    }

    //todo keep unique ones
    public void addDuplicates(Set<String> set){
        allDuplicates.addAll(set);

    }

    public void detect(){
        Map<Integer, Set<String>> hashToIds = new ConcurrentHashMap<>();
        int numDocs = esIndex.getNumDocs();
        IntStream.range(0,numDocs).parallel()
                .filter(i -> esIndex.getStringField("" + i, splitField).
                        equalsIgnoreCase("train"))
                .forEach(i -> {
                    Map<Integer, String> termVector = esIndex.getTermVectorFromIndex("" + i);
                    int hash = termVector.hashCode();
                    if (!hashToIds.containsKey(hash)) {
                        hashToIds.put(hash, Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>()));
                    }
                    hashToIds.get(hash).add("" + i);
                });
        hashToIds.entrySet().stream().parallel().map(Map.Entry::getValue)
                .forEach(this::check);

    }

    private void check(Set<String> candidates){

        Set<Doc> docs = new HashSet<>();
        int size = candidates.size();
        if (size==1){
            return;
        }
        for (String id: candidates){
            Doc doc = new Doc(id,esIndex.getTermVectorFromIndex(id));
            docs.add(doc);
        }
        Set<String> uniqueIds = docs.stream().map(Doc::getId).collect(Collectors.toSet());
        Set<String> candidatesCopy = new HashSet<>(candidates);
        candidatesCopy.removeAll(uniqueIds);
        addDuplicates(candidatesCopy);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        allDuplicates.stream().sorted().forEach(str -> {
            sb.append(str);
            sb.append(",");
        });
        return sb.toString();
    }

    static class Doc{
        private String id;
        private Map<Integer, String> termVector;

        Doc(String id, Map<Integer, String> termVector) {
            this.id = id;
            this.termVector = termVector;
        }


        public String getId() {
            return id;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Doc doc = (Doc) o;

            if (!termVector.equals(doc.termVector)) return false;

            return true;
        }

        @Override
        public int hashCode() {
            return termVector.hashCode();
        }
    }
}
