package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by Ying on 10/1/14.
 */
public class LabelGraph implements Serializable {
    //todo
    /**
     * parent-node hierarchy children-list
     * node children-exclusive none
     * node exclusive list
     * none destination list
     *
     */

    int numLabels;
    private Map<Integer, String> intExtLabelMap;
    private Map<String, Integer> extIntLabelMap;
    private Map<Integer, Set<Integer>> DirectedEdgeMap;
    private Map<Integer, Set<Integer>> InversedDirectedEdgeMap;
    private Map<Integer, Set<Integer>> UndirectedEdgeMap;
    public LabelGraph(int numLabels) {
        this.numLabels = numLabels;
        this.intExtLabelMap = new HashMap<Integer, String>();
        this.extIntLabelMap = new HashMap<String, Integer>();
        this.DirectedEdgeMap = new HashMap<Integer, Set<Integer>>();
        this.InversedDirectedEdgeMap = new HashMap<Integer, Set<Integer>>();
        this.UndirectedEdgeMap = new HashMap<Integer, Set<Integer>>();
    }

    public String getExtLabel(int intLabel) {
        String extLabel = null;
        if (intExtLabelMap.containsKey(intLabel)) {
            extLabel = intExtLabelMap.get(intLabel);
        }
        return extLabel;
    }

    public int getIntLabel(String extLabel) {
        int intLabel = -1;
        if (extIntLabelMap.containsKey(extLabel)) {
            intLabel = extIntLabelMap.get(extLabel);
        }
        return intLabel;
    }

    public void setHierarchyEdge(int srcLabel, int destLabel) {
        Set<Integer> edges1;
        Set<Integer> edges2;
        if (DirectedEdgeMap.containsKey(srcLabel)) {
            edges1 = DirectedEdgeMap.get(srcLabel);
        }
        else {
            edges1 = new HashSet<Integer>();
        }
        if (InversedDirectedEdgeMap.containsKey(destLabel)) {
            edges2 = InversedDirectedEdgeMap.get(destLabel);
        }
        else {
            edges2 = new HashSet<Integer>();
        }
        edges1.add(destLabel);
        edges2.add(srcLabel);
        DirectedEdgeMap.put(srcLabel, edges1);
        InversedDirectedEdgeMap.put(destLabel, edges2);
    }

    public void setExclusionEdge(int srcLabel, int destLabel) {
        if (srcLabel != destLabel) {
            Set<Integer> edges1;
            Set<Integer> edges2;
            if (UndirectedEdgeMap.containsKey(srcLabel)) {
                edges1 = UndirectedEdgeMap.get(srcLabel);
            }
            else {
                edges1 = new HashSet<Integer>();
            }
            if (UndirectedEdgeMap.containsKey(destLabel)) {
                edges2 = UndirectedEdgeMap.get(destLabel);
            }
            else {
                edges2 = new HashSet<Integer>();
            }
            edges1.add(destLabel);
            edges2.add(srcLabel);
            UndirectedEdgeMap.put(srcLabel, edges1);
            UndirectedEdgeMap.put(destLabel, edges2);
        }
    }

    public Set<Integer> getAncestorLabels(int label) {
        Set<Integer> ancestors = new HashSet<Integer>();
        ancestors = getAncestorLabelsHelper(label, ancestors);
        return ancestors;
    }

    public Set<Integer> getAncestorLabelsHelper(int label, Set<Integer> ancestors) {
        if (InversedDirectedEdgeMap.containsKey(label)) {
            for (int i : InversedDirectedEdgeMap.get(label)) {
                if (!ancestors.contains(i)) {
                    ancestors.add(i);
                    ancestors = getAncestorLabelsHelper(i, ancestors);
                }
            }
        }
        return ancestors;
    }

    public Set<Integer> getDescendantLabels(int label) {
        Set<Integer> descendants = new HashSet<Integer>();
        descendants = getDescendantLabelsHelper(label, descendants);
        return descendants;
    }

    public Set<Integer> getDescendantLabelsHelper(int label, Set<Integer> descendants) {
        if (DirectedEdgeMap.containsKey(label)) {
            for (int i : DirectedEdgeMap.get(label)) {
                if (!descendants.contains(i)) {
                    descendants.add(i);
                    descendants = getDescendantLabelsHelper(i, descendants);
                }
            }
        }
        return descendants;
    }

    public Set<Integer> getExclusiveLabels(int label) {
        Set<Integer> exclusions = new HashSet<Integer>();
        exclusions = getExclusiveLabelsHelper(label, exclusions);
        exclusions.remove(label);
        return exclusions;
    }

    public Set<Integer> getExclusiveLabelsHelper(int label, Set<Integer> exclusions) {
        if (UndirectedEdgeMap.containsKey(label)) {
            for (int i : UndirectedEdgeMap.get(label)) {
                if (!exclusions.contains(i)) {
                    exclusions.add(i);
                    exclusions = getExclusiveLabelsHelper(i, exclusions);
                }
            }
        }
        return exclusions;
    }

    /*
    public Set<Integer> getOverlappingLabels(int label) {
        Set<Integer> overlappingLabels = new HashSet<Integer>();
        for (int i = 0; i < numLabels; i++) {
            if (i != label) {
                overlappingLabels.add(i);
            }
        }
        overlappingLabels.removeAll(getAncestorLabels(label));
        overlappingLabels.removeAll(getDescendantLabels(label));
        overlappingLabels.removeAll(getExclusiveLabels(label));
        for (int i : getAncestorLabels(label)) {
            overlappingLabels.removeAll(getExclusiveLabels(i));
        }
        for (int i : getDescendantLabels(label)) {
            overlappingLabels.removeAll(getExclusiveLabels(i));
        }
        return overlappingLabels;
    }*/

    public boolean isConsistent() {
        for (int i = 0; i < numLabels; i++) {
            Set<Integer> ancestors = getAncestorLabels(i);
            ancestors.add(i);
            for (int j : ancestors) {
                Set<Integer> exclusions = getExclusiveLabels(j);
                for (int k : ancestors) {
                    if (exclusions.contains(k)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public boolean isHierarchySubGraphDag() {
        Stack<Integer> dagStack = new Stack<Integer>();
        Set<Integer> uncheckedLabels = new HashSet<Integer>();
        for (int i = 0; i < numLabels; i++) {
            uncheckedLabels.add(i);
        }
        while (!uncheckedLabels.isEmpty()) {
            Iterator<Integer> iter = uncheckedLabels.iterator();
            if (!isDag(iter.next(), dagStack, uncheckedLabels)) {
                return false;
            }
        }
        return true;
    }

    public boolean isDag(int label, Stack<Integer> dagStack, Set<Integer> uncheckedLabels) {
        uncheckedLabels.remove(label);
        if (dagStack.contains(label)) {
            return false;
        }
        dagStack.push(label);
        Set<Integer> descendants = DirectedEdgeMap.get(label);
        if (DirectedEdgeMap.containsKey(label)) {
            for (int i : descendants) {
                if (!isDag(i, dagStack, uncheckedLabels)) {
                    return false;
                }
            }
        }
        dagStack.pop();
        return true;
    }

    public boolean isAssignmentLegal(MultiLabel assignment) {
        for (int i = 0; i < numLabels; i++) {
            Set<Integer> descendants = getDescendantLabels(i);
            if (DirectedEdgeMap.containsKey(i)) {
                if (assignment.matchClass(i) == false) {
                    for (int j : descendants) {
                        if (assignment.matchClass(j) == true) {
                            return false;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < numLabels; i++) {
            Set<Integer> exclusions = getExclusiveLabels(i);
            if (UndirectedEdgeMap.containsKey(i)) {
                if (assignment.matchClass(i) == true) {
                    for (int j : exclusions) {
                        if (assignment.matchClass(j) == true) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    public List<MultiLabel> getLegalAssignments() {
        List<MultiLabel> assignments = new ArrayList<MultiLabel>();
        for (int i = 1; i < Math.pow(2, numLabels); i++) {
            String biLabel = Integer.toBinaryString(i);
            MultiLabel assignment = new MultiLabel(numLabels);
            int j = numLabels - biLabel.length();
            for (int k = numLabels - biLabel.length(); j < numLabels; j++, k++) {
                if (biLabel.charAt(j - numLabels + biLabel.length()) == '1') {
                    assignment.addLabel(k);
                }
            }
            if (isAssignmentLegal(assignment)) {
                assignments.add(assignment);
            }
        }
        return assignments;
    }

    public void parser(String operator) {
        String[] words = operator.split(" ");
        Pattern pattern = Pattern.compile("^[-\\+]?[\\d]*$");
        if (words.length == 2) {
            if (pattern.matcher(words[0]).matches()) {
                //children_exclusive
                int parent = Integer.parseInt(words[0]);
                Set<Integer> children = DirectedEdgeMap.get(parent);
                for (int i : children) {
                    for (int j : children) {
                        if (i < j) {
                            setExclusionEdge(i, j);
                        }
                    }
                }
            }
            else {
                //destination??
                String dest = words[1];
                String[] dests = dest.split(",");
                for (int i = 0; i < dests.length; i++) {
                    DirectedEdgeMap.put(Integer.parseInt(dests[i]), new HashSet<Integer>());
                }
            }
        }
        else {
            if (words[1].contains("hier")) {
                //hierarchy
                int parent = Integer.parseInt(words[0]);
                String[] children = words[2].split(",");
                for (int i = 0; i < children.length; i++) {
                    setHierarchyEdge(parent, Integer.parseInt(children[i]));
                }
            }
            else {
                //exclusive
                int left = Integer.parseInt(words[0]);
                String[] rights = words[2].split(",");
                for (int i = 0; i < rights.length; i++) {
                    setExclusionEdge(left, Integer.parseInt(rights[i]));
                }
            }
        }
    }

}