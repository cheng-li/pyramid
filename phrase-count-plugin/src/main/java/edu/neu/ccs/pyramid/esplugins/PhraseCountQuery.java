package edu.neu.ccs.pyramid.esplugins;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.spans.NearSpansOrdered;
import org.apache.lucene.search.spans.NearSpansUnordered;
import org.apache.lucene.search.spans.SpanCollector;
import org.apache.lucene.search.spans.Spans;

import java.io.IOException;
import java.util.*;


/** Matches spans which are near one another.  One can specify <i>slop</i>, the
 * maximum number of intervening unmatched positions, as well as whether
 * matches are required to be in-order.
 */
public class PhraseCountQuery extends CustomSpanQuery implements Cloneable {

    protected List<CustomSpanTermQuery> clauses;
    protected int slop;
    protected boolean inOrder;
    protected boolean weightedCount;

    protected String field;

    /** Construct a PhraseCountQuery.  Matches spans matching a span from each
     * clause, with up to <code>slop</code> total unmatched positions between
     * them.
     * <br>When <code>inOrder</code> is true, the spans from each clause
     * must be in the same order as in <code>clauses</code> and must be non-overlapping.
     * <br>When <code>inOrder</code> is false, the spans from each clause
     * need not be ordered and may overlap.
     * @param clausesIn the clauses to find near each other, in the same field, at least 2.
     * @param slop The slop value
     * @param inOrder true if order is important
     */
    public PhraseCountQuery(CustomSpanTermQuery[] clausesIn, int slop, boolean inOrder, boolean weightedCount) {
        this.clauses = new ArrayList<>(clausesIn.length);
        for (CustomSpanTermQuery clause : clausesIn) {
            if (this.field == null) {                               // check field
                this.field = clause.getField();
            }
//            else if (clause.getField() != null && !clause.getField().equals(field)) {
//                throw new IllegalArgumentException("Clauses must have same field.");
//            }
            this.clauses.add(clause);
        }
        this.slop = slop;
        this.inOrder = inOrder;
        this.weightedCount = weightedCount;
    }

    /** Return the clauses whose spans are matched. */
    public CustomSpanQuery[] getClauses() {
        return clauses.toArray(new CustomSpanQuery[clauses.size()]);
    }

    /** Return the maximum number of intervening unmatched positions permitted.*/
    public int getSlop() { return slop; }

    /** Return true if matches are required to be in-order.*/
    public boolean isInOrder() { return inOrder; }

    @Override
    public String getField() { return field; }

    @Override
    public String toString(String field) {
        StringBuilder buffer = new StringBuilder();
        buffer.append("phraseCount([");
        Iterator<CustomSpanTermQuery> i = clauses.iterator();
        while (i.hasNext()) {
            CustomSpanQuery clause = i.next();
            buffer.append(clause.toString(field));
            if (i.hasNext()) {
                buffer.append(", ");
            }
        }
        buffer.append("], ");
        buffer.append(slop);
        buffer.append(", ");
        buffer.append(inOrder);
        buffer.append(", ");
        buffer.append(weightedCount);
        buffer.append(")");
        return buffer.toString();
    }

    @Override
    public CustomSpanWeight createWeight(IndexSearcher searcher, boolean needsScores) throws IOException {
        List<CustomSpanWeight> subWeights = new ArrayList<>();
        for (CustomSpanQuery q : clauses) {
            subWeights.add(q.createWeight(searcher, false));
        }
        CustomSpanNearWeight res = new CustomSpanNearWeight(subWeights, searcher, needsScores ? getTermContexts(subWeights) : null);
        res.setWeightedCount(weightedCount);
        return res;
    }

    public class CustomSpanNearWeight extends CustomSpanWeight {

        final List<CustomSpanWeight> subWeights;

        public CustomSpanNearWeight(List<CustomSpanWeight> subWeights, IndexSearcher searcher, Map<Term, TermContext> terms) throws IOException {
            super(PhraseCountQuery.this, searcher, terms);
            this.subWeights = subWeights;
        }

        @Override
        public void extractTermContexts(Map<Term, TermContext> contexts) {
            for (CustomSpanWeight w : subWeights) {
                w.extractTermContexts(contexts);
            }
        }

        @Override
        public Spans getSpans(final LeafReaderContext context, Postings requiredPostings) throws IOException {

            Terms terms = context.reader().terms(field);
            if (terms == null) {
                return null; // field does not exist
            }

            ArrayList<Spans> subSpans = new ArrayList<>(clauses.size());
            for (CustomSpanWeight w : subWeights) {
                Spans subSpan = w.getSpans(context, requiredPostings);
                if (subSpan != null) {
                    subSpans.add(subSpan);
                } else {
                    return null; // all required
                }
            }

            // all NearSpans require at least two subSpans
            return (!inOrder) ? new CustomNearSpansUnordered(slop, subSpans)
                    : new NearSpansOrdered(slop, subSpans);
        }

        @Override
        public void extractTerms(Set<Term> terms) {
            for (CustomSpanWeight w : subWeights) {
                w.extractTerms(terms);
            }
        }
    }

    @Override
    public Query rewrite(IndexReader reader) throws IOException {
        if (clauses.size() < 2) {
            return new TermCountQuery(clauses.get(0).getTerm());
        }
        boolean actuallyRewritten = false;
        List<CustomSpanTermQuery> rewrittenClauses = new ArrayList<>();
        for (int i = 0 ; i < clauses.size(); i++) {
            CustomSpanTermQuery c = clauses.get(i);
            CustomSpanTermQuery query = (CustomSpanTermQuery) c.rewrite(reader);
            actuallyRewritten |= query != c;
            rewrittenClauses.add(query);
        }
        if (actuallyRewritten) {
            try {
                PhraseCountQuery rewritten = (PhraseCountQuery) clone();
                rewritten.clauses = rewrittenClauses;
                return rewritten;
            } catch (CloneNotSupportedException e) {
                throw new AssertionError(e);
            }
        }
        return super.rewrite(reader);
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) &&
                equalsTo(getClass().cast(other));
    }

    private boolean equalsTo(PhraseCountQuery other) {
        return inOrder == other.inOrder &&
                slop == other.slop &&
                clauses.equals(other.clauses);
    }

    @Override
    public int hashCode() {
        int result = classHash();
        result ^= clauses.hashCode();
        result += slop;
        int fac = 1 + (inOrder ? 8 : 4);
        return fac * result;
    }

    private static class CustomSpanGapQuery extends CustomSpanQuery {

        private final String field;
        private final int width;

        public CustomSpanGapQuery(String field, int width) {
            this.field = field;
            this.width = width;
        }

        @Override
        public String getField() {
            return field;
        }

        @Override
        public String toString(String field) {
            return "SpanGap(" + field + ":" + width + ")";
        }

        @Override
        public CustomSpanWeight createWeight(IndexSearcher searcher, boolean needsScores) throws IOException {
            return new CustomSpanGapWeight(searcher);
        }

        private class CustomSpanGapWeight extends CustomSpanWeight {

            CustomSpanGapWeight(IndexSearcher searcher) throws IOException {
                super(CustomSpanGapQuery.this, searcher, null);
            }

            @Override
            public void extractTermContexts(Map<Term, TermContext> contexts) {

            }

            @Override
            public Spans getSpans(LeafReaderContext ctx, Postings requiredPostings) throws IOException {
                return new GapSpans(width);
            }

            @Override
            public void extractTerms(Set<Term> terms) {

            }
        }

        @Override
        public boolean equals(Object other) {
            return sameClassAs(other) &&
                    equalsTo(getClass().cast(other));
        }

        private boolean equalsTo(CustomSpanGapQuery other) {
            return width == other.width &&
                    field.equals(other.field);
        }

        @Override
        public int hashCode() {
            int result = classHash();
            result -= 7 * width;
            return result * 15 - field.hashCode();
        }

    }

    static class GapSpans extends Spans {

        int doc = -1;
        int pos = -1;
        final int width;

        GapSpans(int width) {
            this.width = width;
        }

        @Override
        public int nextStartPosition() throws IOException {
            return ++pos;
        }

        public int skipToPosition(int position) throws IOException {
            return pos = position;
        }

        @Override
        public int startPosition() {
            return pos;
        }

        @Override
        public int endPosition() {
            return pos + width;
        }

        @Override
        public int width() {
            return width;
        }

        @Override
        public void collect(SpanCollector collector) throws IOException {

        }

        @Override
        public int docID() {
            return doc;
        }

        @Override
        public int nextDoc() throws IOException {
            pos = -1;
            return ++doc;
        }

        @Override
        public int advance(int target) throws IOException {
            pos = -1;
            return doc = target;
        }

        @Override
        public long cost() {
            return 0;
        }

        @Override
        public float positionsCost() {
            return 0;
        }
    }

}
