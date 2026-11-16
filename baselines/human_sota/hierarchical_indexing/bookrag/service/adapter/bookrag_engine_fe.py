"""Fixed-evidence retrieval path for the existing BookRAG adaptation.

This deliberately stops after BookRAG's section selection, graph/PPR and
Skyline stages.  It does not run the Map or Reduce answer operators.
"""
from engine import *  # noqa: F401,F403 - reuse the audited BookRAG adaptation


class FixedEvidenceEngine(Engine):
    def prepare(self, question, meta):
        """Run BookRAG planning, falling back to retrieval for malformed global plans.

        The upstream planner occasionally classifies a question as global while
        omitting the structural filters required by PlanResult.  Treating that
        response as an ordinary retrieval query preserves the question and avoids
        fabricating a filter or operation.
        """
        planner = TaskPlanner(self.llm)
        prompt = planner._get_classification_prompt(question) + '''\nCombine classification, operator planning, and query-entity extraction into ONE JSON response to respect the benchmark budget.
For complex questions, return at most 3 independent retrieval sub_questions and a synthesis step. Do not invent missing entity names. For simple questions include exactly one retrieval sub_question equal to the original query and leave filters empty; it will be ignored after classification. For global questions return explicit structural filters and operation COUNT/LIST/SUMMARIZE/ANALYZE. Extract only named concepts present in the question. Entity types: ''' + ', '.join(QUESTION_ENTITY_TYPES)
        prompt += '\nThe following decomposition examples apply ONLY if classification is complex. They do not override classification.\n' + planner._get_complex_prompt(question) + '\nFirst classify the original query using the three category definitions above. Apply decomposition only if complex. Return the combined QueryPlan schema, with at most 8 entities, rather than the example response schema.'
        raw = self.llm.get_json_completion(prompt, QueryPlan)
        if raw.query_type not in ('simple', 'complex', 'global'):
            raise ValueError('Invalid query type')
        self._planner_fallback = None
        if raw.query_type == 'complex':
            plan = PlanResult(query_type='complex', original_query=question,
                              sub_questions=[SubQuestion(**q.model_dump()) for q in raw.sub_questions])
        elif raw.query_type == 'global' and raw.filters and raw.operation:
            plan = PlanResult(query_type='global', original_query=question,
                              filters=raw.filters, operation=raw.operation)
        else:
            if raw.query_type == 'global':
                self._planner_fallback = 'global_without_required_filters_or_operation'
            plan = PlanResult(query_type='simple', original_query=question)
        return raw, plan

    def retrieve_passages(self, question, meta):
        started = time.time()
        self.llm.calls = 0
        self.retriever.graph_calls = 0
        self.retriever.text_calls = 0
        self.retriever.skyline_calls = 0
        doc_id = (meta or {}).get("doc_id")
        raw, plan = self.prepare(question, meta or {})
        trace = {"question": question, "meta": meta or {}, "plan": plan.model_dump(), "steps": []}
        if self._planner_fallback:
            trace["planner_fallback"] = self._planner_fallback

        if plan.query_type == "global":
            nodes = filter_tree_nodes(self.tree, plan.filters)
            if doc_id:
                nodes = [n for n in nodes if self.node_doc.get(n.index_id) == doc_id]
            evidence = [n for n in nodes if n.type == NodeType.TEXT]
            trace["global_filtered_ids"] = [n.index_id for n in nodes]
        else:
            queries = [s.question for s in (plan.sub_questions or []) if s.type == "retrieval"] or [question]
            queries = queries[:3]
            sections = [n for n in self.sections if not doc_id or self.node_doc.get(n.index_id) == doc_id]
            all_ids = [self.section_pos[n.index_id] for n in sections]
            valid_section_ids = {n.index_id for n in sections}
            selection, steps = [], []
            for k, query in enumerate(queries):
                mapped = self.link(raw.entities, query, doc_id)
                linked = set()
                for entities in mapped.values():
                    for entity in entities:
                        for node_id in self.graph.NodeName2TreeNodes(entity):
                            if doc_id and self.node_doc.get(node_id) != doc_id:
                                continue
                            ancestor = self.tree.get_ancestor_at_depth(node_id, 2 if doc_id else 1)
                            if ancestor and ancestor.index_id in valid_section_ids:
                                linked.add(ancestor.index_id)
                qv = self.embedder.embed_texts([query])[0]
                order = np.argsort(-(self.section_vec[all_ids] @ qv))
                supplemental = [sections[int(j)] for j in order[:24]
                                if sections[int(j)].index_id not in linked]
                linked_nodes = [n for n in sections if n.index_id in linked]
                if len(linked_nodes) > 24:
                    scores = self.reranker.rerank(query, [self.section_text(n) for n in linked_nodes])
                    linked_nodes = [linked_nodes[j] for j in np.argsort(-np.asarray(scores))[:24]]
                    linked = {n.index_id for n in linked_nodes}
                selection.append({
                    "step": k,
                    "query": query,
                    "primary_candidates": [self.section_info(n) for n in linked_nodes],
                    "remaining_sections": [self.section_info(n) for n in supplemental],
                })
                steps.append({"query": query, "mapped": mapped, "linked": sorted(linked),
                              "valid": {n.index_id for n in supplemental}})

            choices = self.llm.get_json_completion(
                "You are the BookRAG section selector. Keep the primary sections linked by query entities. "
                "Select supplementary section IDs only when they add evidence needed by the query. Use only "
                "the listed remaining IDs; select at most 5 per step. Return choices for every step.\n"
                + json.dumps(selection, ensure_ascii=False), Choices)
            ordered = {}
            for k, step in enumerate(steps):
                extra = [i for c in choices.choices if c.step == k for i in c.supplementary_ids
                         if i in step["valid"]][:5]
                selected = sorted(set(step["linked"] + extra))
                subtree = self.tree.get_subtree_nodes(selected)
                subtree = [n for n in subtree if n.type == NodeType.TEXT and
                           (not doc_id or self.node_doc.get(n.index_id) == doc_id)]
                graph = self.graph.get_kg_subgraph([n.index_id for n in subtree])
                ids, entities = self.retriever.skyline_filter(step["query"], subtree, graph, step["mapped"])
                if not ids:
                    raise RuntimeError("BookRAG graph retrieval produced no evidence; no vector-RAG fallback")
                for node_id in ids:
                    ordered.setdefault(node_id, self.tree.nodes[node_id])
                trace["steps"].append({"query": step["query"], "entity_map": step["mapped"],
                    "linked_sections": step["linked"], "supplementary_sections": extra,
                    "selected_sections": selected, "subgraph_entities": len(graph),
                    "subgraph_edges": graph.number_of_edges(), "ppr_entities": entities,
                    "retrieved_node_ids": ids})
            evidence = list(ordered.values())

        passages, seen = [], set()
        for node in evidence:
            did = str(self.node_doc[node.index_id])
            text = node.meta_info.content
            key_ = (did, text)
            if not text or key_ in seen:
                continue
            seen.add(key_)
            passages.append({"doc_id": did, "title": self.docs[did].get("title", "") or "", "text": text})
            if len(passages) == 256:
                break
        trace["core"] = {"classification": True, "planning": True,
                         "graph_reasoning": bool(trace.get("steps")),
                         "skyline": self.retriever.skyline_calls,
                         "graph_calls": self.retriever.graph_calls,
                         "text_reranker_calls": self.retriever.text_calls,
                         "llm_calls": self.llm.calls, "elapsed": time.time() - started,
                         "passages": len(passages)}
        if not passages:
            raise RuntimeError("BookRAG returned empty evidence")
        return passages, trace
