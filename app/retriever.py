import re

from app.embeddings import create_query_embedding


class Retriever:

    def __init__(
        self,
        vector_store,
        metadata,
        similarity_threshold=0.20
    ):
        self.vector_store = vector_store
        self.metadata = metadata
        self.similarity_threshold = similarity_threshold

    # =========================================================
    # NORMALIZE
    # =========================================================

    def _normalize_result(self, result):

        if isinstance(result, dict):
            return result

        return None

    # =========================================================
    # DUPLICATES
    # =========================================================

    def _remove_duplicates(self, results):

        unique = []
        seen = set()

        for result in results:

            if not result:
                continue

            source = result.get("source", "")
            page = result.get("page", "")
            image_index = result.get("image_index", "")
            chunk_type = result.get("type", "text")
            text = result.get("text", "")

            if chunk_type == "image":

                key = (
                    source,
                    page,
                    image_index
                )

            else:

                key = (
                    source,
                    page,
                    text[:300]
                )

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

        return unique

    # =========================================================
    # VISUAL QUERY
    # =========================================================

    def _is_visual_query(self, query):

        query = query.lower()

        visual_terms = [
            "figure",
            "fig.",
            "diagram",
            "image",
            "picture",
            "chart",
            "graph",
            "table",
            "flowchart",
            "illustration",
            "illustrates",
            "shown",
            "shows",
            "what does the figure",
            "what does figure",
            "what does the diagram",
            "what does the image",
            "what is shown",
            "what is illustrated",
            "explain the figure",
            "explain figure",
            "explain the diagram",
            "explain the image"
        ]

        return any(
            term in query
            for term in visual_terms
        )

    # =========================================================
    # EXPLICIT FIGURE NUMBER
    # =========================================================

    def _extract_figure_number(self, query):

        match = re.search(
            r"(?:figure|fig\.?)\s*(\d+(?:\.\d+)?)",
            query.lower()
        )

        if match:
            return match.group(1)

        return None

    # =========================================================
    # KEYWORDS
    # =========================================================

    def _extract_keywords(self, query):

        query = query.lower()

        # Remove common question words.
        query = re.sub(
            r"\b(what|is|are|was|were|does|do|did|"
            r"the|a|an|how|why|where|which|"
            r"explain|show|shown|mean|means)\b",
            " ",
            query
        )

        words = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            query
        )

        # Keep useful technical terms.
        keywords = []

        for word in words:

            if len(word) >= 3:
                keywords.append(word)

        return list(dict.fromkeys(keywords))

    # =========================================================
    # KEYWORD SCORE
    # =========================================================

    def _keyword_score(self, query, result):

        text = str(
            result.get("text", "")
        ).lower()

        if not text:
            return 0.0

        keywords = self._extract_keywords(query)

        if not keywords:
            return 0.0

        matched = 0

        for keyword in keywords:

            if keyword in text:
                matched += 1

        return matched / len(keywords)

    # =========================================================
    # TECHNICAL TERM DETECTION
    # =========================================================

    def _technical_terms(self, query):

        terms = []

        # MPI functions / constants
        mpi_terms = re.findall(
            r"\bMPI_[A-Za-z0-9_]+\b",
            query
        )

        terms.extend(mpi_terms)

        # Figure numbers
        figure_number = self._extract_figure_number(query)

        if figure_number:
            terms.append(
                f"figure {figure_number}"
            )

        return [
            term.lower()
            for term in terms
        ]

    # =========================================================
    # TECHNICAL MATCH
    # =========================================================

    def _technical_match_score(self, query, result):

        text = str(
            result.get("text", "")
        ).lower()

        if not text:
            return 0.0

        terms = self._technical_terms(query)

        if not terms:
            return 0.0

        matches = 0

        for term in terms:

            if term in text:
                matches += 1

        return matches / len(terms)

    # =========================================================
    # EXACT FIGURE SEARCH
    # =========================================================

    def _find_exact_figure(self, query):

        figure_number = self._extract_figure_number(query)

        if not figure_number:
            return []

        target = (
            f"figure {figure_number}"
        ).lower()

        matches = []

        for index, item in enumerate(self.metadata):

            if not isinstance(item, dict):
                continue

            text = str(
                item.get("text", "")
            ).lower()

            if target not in text:
                continue

            # Prefer actual image chunks.
            if item.get("type") == "image":

                result = dict(item)

                result["index"] = index
                result["score"] = 1.0

                matches.append(result)

        return matches

    # =========================================================
    # PAGE CONTEXT
    # =========================================================

    def _get_page_context(
        self,
        source,
        page,
        max_items=8
    ):

        try:
            page = int(page)
        except (TypeError, ValueError):
            return []

        candidates = []

        for index, item in enumerate(self.metadata):

            if not isinstance(item, dict):
                continue

            if item.get("source") != source:
                continue

            try:
                item_page = int(
                    item.get("page")
                )
            except (TypeError, ValueError):
                continue

            # Same page + nearby page.
            distance = abs(
                item_page - page
            )

            if distance <= 1:

                result = dict(item)

                result["index"] = index

                # Context results receive a modest score.
                result["score"] = max(
                    0.25,
                    0.45 - (distance * 0.10)
                )

                candidates.append(
                    result
                )

        candidates.sort(
            key=lambda x: (
                -self._technical_match_score(
                    "",
                    x
                ),
                -self._keyword_score(
                    "",
                    x
                )
            )
        )

        return candidates[:max_items]

    # =========================================================
    # APPLY RELEVANCE BOOST
    # =========================================================

    def _apply_relevance_boost(
        self,
        query,
        results
    ):

        boosted = []

        for result in results:

            faiss_score = float(
                result.get("score", 0.0)
            )

            keyword_score = (
                self._keyword_score(
                    query,
                    result
                )
            )

            technical_score = (
                self._technical_match_score(
                    query,
                    result
                )
            )

            # Base FAISS score.
            final_score = faiss_score

            # General keyword evidence.
            final_score += (
                keyword_score * 0.15
            )

            # Strong boost for exact technical terms
            # such as MPI_Ssend.
            final_score += (
                technical_score * 0.35
            )

            result["faiss_score"] = faiss_score
            result["keyword_score"] = keyword_score
            result["technical_score"] = technical_score
            result["score"] = final_score

            boosted.append(result)

        return boosted

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query,
        top_k=5
    ):

        print(
            f"Retriever: Query = {query}",
            flush=True
        )

        visual = self._is_visual_query(
            query
        )

        if visual:

            print(
                "Retriever: Visual question detected.",
                flush=True
            )

        # -----------------------------------------------------
        # EXACT FIGURE
        # -----------------------------------------------------

        exact_figure_results = []

        if visual:

            figure_number = (
                self._extract_figure_number(
                    query
                )
            )

            if figure_number:

                print(
                    f"Retriever: Explicit Figure "
                    f"{figure_number} detected.",
                    flush=True
                )

                exact_figure_results = (
                    self._find_exact_figure(
                        query
                    )
                )

                print(
                    "Retriever: Direct metadata search "
                    f"found {len(exact_figure_results)} "
                    "match(es).",
                    flush=True
                )

        # -----------------------------------------------------
        # EMBEDDING SEARCH
        # -----------------------------------------------------

        query_embedding = (
            create_query_embedding(
                query
            )
        )

        search_k = min(
            max(top_k * 10, 50),
            self.vector_store.size()
        )

        raw_results = (
            self.vector_store.search(
                query_embedding,
                search_k
            )
        )

        results = []

        # -----------------------------------------------------
        # CONVERT FAISS RESULTS
        # -----------------------------------------------------

        for item in raw_results:

            if not isinstance(item, dict):
                continue

            index = item.get("index")

            if index is None:
                continue

            try:
                index = int(index)
            except (TypeError, ValueError):
                continue

            if index < 0:
                continue

            if index >= len(self.metadata):
                continue

            metadata_item = self.metadata[index]

            if not isinstance(metadata_item, dict):
                continue

            result = dict(
                metadata_item
            )

            result["index"] = index
            result["score"] = float(
                item.get(
                    "score",
                    0.0
                )
            )

            results.append(
                result
            )

        print(
            f"Retriever: Converted "
            f"{len(results)} metadata results.",
            flush=True
        )

        # -----------------------------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------------------------

        results = self._remove_duplicates(
            results
        )

        # -----------------------------------------------------
        # RELEVANCE BOOST
        # -----------------------------------------------------

        results = self._apply_relevance_boost(
            query,
            results
        )

        results.sort(
            key=lambda x: x.get(
                "score",
                0.0
            ),
            reverse=True
        )

        # -----------------------------------------------------
        # DEBUG
        # -----------------------------------------------------

        if results:

            print(
                "Retriever: Top candidates:",
                flush=True
            )

            for result in results[:10]:

                print(
                    f"  score={result.get('score', 0):.4f} "
                    f"| faiss={result.get('faiss_score', 0):.4f} "
                    f"| keyword={result.get('keyword_score', 0):.2f} "
                    f"| technical={result.get('technical_score', 0):.2f} "
                    f"| page={result.get('page', '?')} "
                    f"| type={result.get('type', '?')}",
                    flush=True
                )

        # -----------------------------------------------------
        # EXACT FIGURE MODE
        # -----------------------------------------------------

        if exact_figure_results:

            exact = (
                exact_figure_results[0]
            )

            print(
                "Retriever: Exact figure found: "
                f"{exact.get('source', '')} | "
                f"page={exact.get('page', '?')} | "
                f"image={exact.get('image_index', '?')}",
                flush=True
            )

            # Find surrounding text on same page.
            surrounding = []

            source = exact.get(
                "source"
            )

            page = exact.get(
                "page"
            )

            for item in results:

                if (
                    item.get("source") == source
                    and str(item.get("page")) == str(page)
                    and item.get("type") != "image"
                ):

                    surrounding.append(
                        item
                    )

            # If the direct FAISS result didn't contain
            # surrounding text, use metadata directly.
            if not surrounding:

                for index, item in enumerate(
                    self.metadata
                ):

                    if not isinstance(item, dict):
                        continue

                    if (
                        item.get("source") == source
                        and str(item.get("page")) == str(page)
                        and item.get("type") != "image"
                    ):

                        candidate = dict(item)

                        candidate["index"] = index
                        candidate["score"] = 1.0

                        surrounding.append(
                            candidate
                        )

            final = [
                exact
            ]

            if surrounding:

                final.append(
                    surrounding[0]
                )

            print(
                "Retriever: Returning exact Figure.",
                flush=True
            )

            for result in final:

                print(
                    f"  score={result.get('score', 0):.4f} "
                    f"| index={result.get('index', '?')} "
                    f"| page={result.get('page', '?')} "
                    f"| type={result.get('type', '?')}",
                    flush=True
                )

            return final

        # -----------------------------------------------------
        # TECHNICAL QUERY
        # -----------------------------------------------------

        technical_terms = (
            self._technical_terms(query)
        )

        if technical_terms:

            print(
                "Retriever: Technical term detected: "
                + ", ".join(technical_terms),
                flush=True
            )

            # Look through ALL metadata for exact technical
            # term occurrences. This fixes cases such as
            # "What is MPI_Ssend?" where semantic ranking
            # alone may place the explanation too low.
            exact_term_results = []

            for index, item in enumerate(
                self.metadata
            ):

                if not isinstance(item, dict):
                    continue

                text = str(
                    item.get("text", "")
                ).lower()

                matched = False

                for term in technical_terms:

                    if term in text:

                        matched = True
                        break

                if not matched:
                    continue

                result = dict(item)

                result["index"] = index
                result["score"] = 1.0
                result["technical_score"] = 1.0

                exact_term_results.append(
                    result
                )

            exact_term_results = (
                self._remove_duplicates(
                    exact_term_results
                )
            )

            # Put exact technical matches first.
            exact_term_results.sort(
                key=lambda x: (
                    x.get("type") == "image",
                    -len(
                        x.get(
                            "text",
                            ""
                        )
                    )
                )
            )

            # Prefer text over unrelated images.
            text_exact = [
                r for r in exact_term_results
                if r.get("type") != "image"
            ]

            if text_exact:

                results = (
                    text_exact
                    + results
                )

                results = (
                    self._remove_duplicates(
                        results
                    )
                )

                print(
                    "Retriever: Exact technical "
                    "term matches found: "
                    f"{len(text_exact)}",
                    flush=True
                )

        # -----------------------------------------------------
        # THRESHOLD
        # -----------------------------------------------------

        filtered = [
            result
            for result in results
            if result.get(
                "score",
                0.0
            ) >= self.similarity_threshold
        ]

        if not filtered and results:

            print(
                "Retriever: No result passed threshold. "
                "Using best available result.",
                flush=True
            )

            filtered = results[:1]

        # -----------------------------------------------------
        # VISUAL QUESTIONS
        # -----------------------------------------------------

        if visual:

            image_results = [
                result
                for result in filtered
                if result.get("type") == "image"
            ]

            text_results = [
                result
                for result in filtered
                if result.get("type") != "image"
            ]

            image_results = image_results[:2]

            remaining = max(
                top_k - len(image_results),
                1
            )

            filtered = (
                image_results
                + text_results[:remaining]
            )

        # -----------------------------------------------------
        # NORMAL QUESTIONS
        # -----------------------------------------------------

        else:

            text_results = [
                result
                for result in filtered
                if result.get("type") != "image"
            ]

            if text_results:

                filtered = (
                    text_results[:top_k]
                )

            else:

                filtered = (
                    filtered[:top_k]
                )

        # -----------------------------------------------------
        # FINAL SORT
        # -----------------------------------------------------

        filtered.sort(
            key=lambda x: x.get(
                "score",
                0.0
            ),
            reverse=True
        )

        print(
            f"Retriever: Returning "
            f"{len(filtered[:top_k])} result(s).",
            flush=True
        )

        return filtered[:top_k]
