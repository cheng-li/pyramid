package edu.neu.ccs.pyramid.esplugins;

/**
 * Created by maoqiuzi on 5/24/17.
 */
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

import org.apache.lucene.codecs.lucene50.Lucene50PostingsFormat;
import org.apache.lucene.codecs.lucene50.Lucene50PostingsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.util.ArrayUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Set;

/** A Query that matches documents containing a particular sequence of terms.
 * A PhraseCountQuery is built by QueryParser for input like <code>"new york"</code>.
 */
public class PhraseCountQuery extends Query {



    private final int slop;
    private final String field;
    private final Term[] terms;
    private final int[] positions;
    private final boolean weightedCount;

    private PhraseCountQuery(int slop, boolean weightedCount, Term[] terms, int[] positions) {
        if (terms.length != positions.length) {
            throw new IllegalArgumentException("Must have as many terms as positions");
        }
        if (slop < 0) {
            throw new IllegalArgumentException("Slop must be >= 0, got " + slop);
        }
        this.slop = slop;
        this.weightedCount = weightedCount;
        this.terms = terms;
        this.positions = positions;
        this.field = terms.length == 0 ? null : terms[0].field();
    }

    private static int[] incrementalPositions(int length) {
        int[] positions = new int[length];
        for (int i = 0; i < length; ++i) {
            positions[i] = i;
        }
        return positions;
    }

    /**
     * Create a phrase query which will match documents that contain the given
     * list of terms at consecutive positions in {@code field}, and at a
     * maximum edit distance of {@code slop}.
     * @see #getSlop()
     */
    public PhraseCountQuery(int slop, boolean weightedCount, Term... terms) {
        this(slop, weightedCount, terms, incrementalPositions(terms.length));
    }

    /**
     * Return the slop for this {@link PhraseCountQuery}.
     *
     * <p>The slop is an edit distance between respective positions of terms as
     * defined in this {@link PhraseCountQuery} and the positions of terms in a
     * document.
     *
     * <p>For instance, when searching for {@code "quick fox"}, it is expected that
     * the difference between the positions of {@code fox} and {@code quick} is 1.
     * So {@code "a quick brown fox"} would be at an edit distance of 1 since the
     * difference of the positions of {@code fox} and {@code quick} is 2.
     * Similarly, {@code "the fox is quick"} would be at an edit distance of 3
     * since the difference of the positions of {@code fox} and {@code quick} is -2.
     * The slop defines the maximum edit distance for a document to match.
     *
     * <p>More exact matches are scored higher than sloppier matches, thus search
     * results are sorted by exactness.
     */
    public int getSlop() { return slop; }

    /** Returns the list of terms in this phrase. */
    public Term[] getTerms() {
        return terms;
    }

    /**
     * Returns the relative positions of terms in this phrase.
     */
    public int[] getPositions() {
        return positions;
    }

    @Override
    public Query rewrite(IndexReader reader) throws IOException {
        if (terms.length == 0) {
            return new MatchNoDocsQuery("empty PhraseCountQuery");
        } else if (terms.length == 1) {
            return new TermCountQuery(terms[0]);
        } else if (positions[0] != 0) {
            int[] newPositions = new int[positions.length];
            for (int i = 0; i < positions.length; ++i) {
                newPositions[i] = positions[i] - positions[0];
            }
            return new PhraseCountQuery(slop, weightedCount, terms, newPositions);
        } else {
            return super.rewrite(reader);
        }
    }

    static class PostingsAndFreq implements Comparable<PostingsAndFreq> {
        final PostingsEnum postings;
        final int position;
        final Term[] terms;
        final int nTerms; // for faster comparisons

        public PostingsAndFreq(PostingsEnum postings, int position, Term... terms) {
            this.postings = postings;
            this.position = position;
            nTerms = terms==null ? 0 : terms.length;
            if (nTerms>0) {
                if (terms.length==1) {
                    this.terms = terms;
                } else {
                    Term[] terms2 = new Term[terms.length];
                    System.arraycopy(terms, 0, terms2, 0, terms.length);
                    Arrays.sort(terms2);
                    this.terms = terms2;
                }
            } else {
                this.terms = null;
            }
        }

        @Override
        public int compareTo(PhraseCountQuery.PostingsAndFreq other) {
            if (position != other.position) {
                return position - other.position;
            }
            if (nTerms != other.nTerms) {
                return nTerms - other.nTerms;
            }
            if (nTerms == 0) {
                return 0;
            }
            for (int i=0; i<terms.length; i++) {
                int res = terms[i].compareTo(other.terms[i]);
                if (res!=0) return res;
            }
            return 0;
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + position;
            for (int i=0; i<nTerms; i++) {
                result = prime * result + terms[i].hashCode();
            }
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null) return false;
            if (getClass() != obj.getClass()) return false;
            PhraseCountQuery.PostingsAndFreq other = (PhraseCountQuery.PostingsAndFreq) obj;
            if (position != other.position) return false;
            if (terms == null) return other.terms == null;
            return Arrays.equals(terms, other.terms);
        }
    }

    private class PhraseWeight extends Weight {
        private final boolean needsScores;
        private transient TermContext states[];

        public PhraseWeight(IndexSearcher searcher, boolean needsScores)
                throws IOException {
            super(PhraseCountQuery.this);
            final int[] positions = PhraseCountQuery.this.getPositions();
            if (positions.length < 2) {
                throw new IllegalStateException("PhraseWeight does not support less than 2 terms, call rewrite first");
            } else if (positions[0] != 0) {
                throw new IllegalStateException("PhraseWeight requires that the first position is 0, call rewrite first");
            }
            this.needsScores = needsScores;
            final IndexReaderContext context = searcher.getTopReaderContext();
            states = new TermContext[terms.length];
            TermStatistics termStats[] = new TermStatistics[terms.length];
            for (int i = 0; i < terms.length; i++) {
                final Term term = terms[i];
                states[i] = TermContext.build(context, term);
                termStats[i] = searcher.termStatistics(term, states[i]);
            }
        }

        @Override
        public void extractTerms(Set<Term> queryTerms) {
            Collections.addAll(queryTerms, terms);
        }

        @Override
        public String toString() { return "weight(" + PhraseCountQuery.this + ")"; }

        // not used
        @Override
        public float getValueForNormalization() {
            return 1f;
        }

        // not used
        @Override
        public void normalize(float queryNorm, float boost) {
            ;
        }

        @Override
        public Scorer scorer(LeafReaderContext context) throws IOException {
            assert terms.length > 0;
            final LeafReader reader = context.reader();
            PhraseCountQuery.PostingsAndFreq[] postingsFreqs = new PhraseCountQuery.PostingsAndFreq[terms.length];

            final Terms fieldTerms = reader.terms(field);
            if (fieldTerms == null) {
                return null;
            }

            if (fieldTerms.hasPositions() == false) {
                throw new IllegalStateException("field \"" + field + "\" was indexed without position data; cannot run PhraseCountQuery (phrase=" + getQuery() + ")");
            }

            // Reuse single TermsEnum below:
            final TermsEnum te = fieldTerms.iterator();
            float totalMatchCost = 0;

            for (int i = 0; i < terms.length; i++) {
                final Term t = terms[i];
                final TermState state = states[i].get(context.ord);
                if (state == null) { /* term doesnt exist in this segment */
                    assert termNotInReader(reader, t): "no termstate found but term exists in reader";
                    return null;
                }
                te.seekExact(t.bytes(), state);
                PostingsEnum postingsEnum = te.postings(null, PostingsEnum.POSITIONS);
                postingsFreqs[i] = new PhraseCountQuery.PostingsAndFreq(postingsEnum, positions[i], t);
                totalMatchCost += termPositionsCost(te);
            }

            // sort by increasing docFreq order
            if (slop == 0) {
                ArrayUtil.timSort(postingsFreqs);
            }

            return new PhraseCountScorer(this, postingsFreqs, slop,
                    needsScores, weightedCount, totalMatchCost);
        }

        // only called from assert
        private boolean termNotInReader(LeafReader reader, Term term) throws IOException {
            return reader.docFreq(term) == 0;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            Scorer scorer = scorer(context);
            if (scorer != null) {
                int newDoc = scorer.iterator().advance(doc);
                if (newDoc == doc) {
                    if (weightedCount) {
                        return Explanation.match((scorer).score(), "sloppy frequency");
                    }
                    return Explanation.match(scorer.score(), "phrase frequency");
                }
            }

            return Explanation.noMatch("no matching term");
        }
    }

    /** A guess of
     * the average number of simple operations for the initial seek and buffer refill
     * per document for the positions of a term.
     * See also {@link Lucene50PostingsReader.BlockPostingsEnum#nextPosition()}.
     * <p>
     * Aside: Instead of being constant this could depend among others on
     * {@link Lucene50PostingsFormat#BLOCK_SIZE},
     * {@link TermsEnum#docFreq()},
     * {@link TermsEnum#totalTermFreq()},
     * {@link DocIdSetIterator#cost()} (expected number of matching docs),
     * {@link LeafReader#maxDoc()} (total number of docs in the segment),
     * and the seek time and block size of the device storing the index.
     */
    private static final int TERM_POSNS_SEEK_OPS_PER_DOC = 128;

    /** Number of simple operations in {@link Lucene50PostingsReader.BlockPostingsEnum#nextPosition()}
     *  when no seek or buffer refill is done.
     */
    private static final int TERM_OPS_PER_POS = 7;

    /** Returns an expected cost in simple operations
     *  of processing the occurrences of a term
     *  in a document that contains the term.
     *  This is for use by {@link TwoPhaseIterator#matchCost} implementations.
     *  <br>This may be inaccurate when {@link TermsEnum#totalTermFreq()} is not available.
     *  @param termsEnum The term is the term at which this TermsEnum is positioned.
     */
    static float termPositionsCost(TermsEnum termsEnum) throws IOException {
        int docFreq = termsEnum.docFreq();
        assert docFreq > 0;
        long totalTermFreq = termsEnum.totalTermFreq(); // -1 when not available
        float expOccurrencesInMatchingDoc = (totalTermFreq < docFreq) ? 1 : (totalTermFreq / (float) docFreq);
        return TERM_POSNS_SEEK_OPS_PER_DOC + expOccurrencesInMatchingDoc * TERM_OPS_PER_POS;
    }


    @Override
    public Weight createWeight(IndexSearcher searcher, boolean needsScores) throws IOException {
        return new PhraseCountQuery.PhraseWeight(searcher, needsScores);
    }

    /** Prints a user-readable version of this query. */
    @Override
    public String toString(String f) {
        StringBuilder buffer = new StringBuilder();
        if (field != null && !field.equals(f)) {
            buffer.append(field);
            buffer.append(":");
        }

        buffer.append("\"");
        final int maxPosition;
        if (positions.length == 0) {
            maxPosition = -1;
        } else {
            maxPosition = positions[positions.length - 1];
        }
        String[] pieces = new String[maxPosition + 1];
        for (int i = 0; i < terms.length; i++) {
            int pos = positions[i];
            String s = pieces[pos];
            if (s == null) {
                s = (terms[i]).text();
            } else {
                s = s + "|" + (terms[i]).text();
            }
            pieces[pos] = s;
        }
        for (int i = 0; i < pieces.length; i++) {
            if (i > 0) {
                buffer.append(' ');
            }
            String s = pieces[i];
            if (s == null) {
                buffer.append('?');
            } else {
                buffer.append(s);
            }
        }
        buffer.append("\"");

        if (slop != 0) {
            buffer.append("~");
            buffer.append(slop);
        }

        return buffer.toString();
    }

    /** Returns true iff <code>o</code> is equal to this. */
    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) &&
                equalsTo(getClass().cast(other));
    }

    private boolean equalsTo(PhraseCountQuery other) {
        return slop == other.slop &&
                weightedCount == other.weightedCount &&
                Arrays.equals(terms, other.terms) &&
                Arrays.equals(positions, other.positions);
    }

    /** Returns a hash code value for this object.*/
    @Override
    public int hashCode() {
        int h = classHash();
        h = 31 * h + slop;
        h = 31 * h + Arrays.hashCode(terms);
        h = 31 * h + Arrays.hashCode(positions);
        int t = (weightedCount) ? 1 : 0;
        h = 31 * h + t;
        return h;
    }

}

