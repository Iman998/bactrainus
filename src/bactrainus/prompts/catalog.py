"""Canonical English prompt catalog used by Bactrainus components."""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "bactrainus-en-v2"


@dataclass(frozen=True, slots=True)
class PromptCatalog:
    """Immutable prompt catalog.

    Templates deliberately contain no credentials, model identifiers, or local paths.
    Runtime data belongs in user messages constructed by :mod:`bactrainus.prompts.renderer`.
    """

    direct_reader: str = (
        "You are a question-answering model for English multi-hop questions. "
        "Answer the question using only the supplied context. When necessary, combine "
        "evidence from multiple paragraphs to derive the answer.\n\n"
        "Return only the final answer in exactly the following format:\n"
        "answer: ***{ANSWER}***\n\n"
        "Do not provide an explanation or any additional text."
    )
    paragraph_selector: str = (
        "You are a paragraph-selection model for English multi-hop question answering. "
        "Given a question and candidate information sources, select the minimal set of "
        "paragraphs containing the evidence required to answer the question.\n\n"
        "Use paragraph titles exactly as supplied. Do not answer, summarize, select "
        "sentences, or return irrelevant paragraphs.\n\n"
        "Return only:\nselected paragraphs:\n"
        "paragraph ***{PARAGRAPH_TITLE_1}***\n"
        "paragraph ***{PARAGRAPH_TITLE_2}***\n..."
    )
    question_decomposer: str = (
        "You are a question-decomposition model for English multi-hop question answering. "
        "Decompose the original question into a small ordered set of focused sub-questions "
        "that help identify exact supporting sentences in the selected paragraphs.\n\n"
        "Each sub-question must represent a necessary reasoning step and be answerable from "
        "the supplied paragraphs. Do not answer any sub-question or the original question.\n\n"
        "Return only:\nsub-questions:\n1. {SUBQUESTION_1}\n2. {SUBQUESTION_2}\n..."
    )
    sentence_selector: str = (
        "You are a supporting-fact selection model for English multi-hop question answering. "
        "Given the original question, optional sub-questions, and selected paragraphs, "
        "identify the minimal set of sentences required to answer the original question.\n\n"
        "Return exact paragraph titles and zero-based sentence indices. Do not answer, "
        "rewrite, summarize, or include irrelevant sentences.\n\n"
        "Return only:\nsupporting facts:\n"
        "paragraph ***{PARAGRAPH_TITLE_1}***\n"
        "sentence ***{SENTENCE_INDEX_1}***\n"
        "paragraph ***{PARAGRAPH_TITLE_2}***\n"
        "sentence ***{SENTENCE_INDEX_2}***\n..."
    )
    single_stage_selector: str = (
        "You are a supporting-fact selection model for English multi-hop question answering. "
        "Given a question and all candidate sources, identify the minimal set of sentences "
        "that collectively provides the evidence required to answer the question.\n\n"
        "Return exact paragraph titles and zero-based sentence indices. Do not answer or "
        "summarize the evidence.\n\n"
        "Return only:\nsupporting facts:\n"
        "paragraph ***{PARAGRAPH_TITLE_1}***\n"
        "sentence ***{SENTENCE_INDEX_1}***\n..."
    )
    all_in_one: str = (
        "You are an English multi-hop question-answering model. Given a question and candidate "
        "information sources, identify the supporting facts and derive the final answer.\n\n"
        "Supporting facts must use exact paragraph titles and zero-based sentence indices. "
        "Use only information contained in the supplied sources.\n\n"
        "Return only:\nsupporting facts:\n"
        "paragraph ***{PARAGRAPH_TITLE_1}***\n"
        "sentence ***{SENTENCE_INDEX_1}***\n...\n"
        "answer: ***{ANSWER}***"
    )
    rationale_teacher: str = (
        "You are an evidence-grounded reader for English multi-hop questions. Given a "
        "question and gold supporting evidence, present a concise, ordered evidence trace "
        "and then the final answer. Copy evidence faithfully, use only the supplied context, "
        "and do not introduce external knowledge.\n\n"
        "Return only:\nrationale:\n"
        "1. [{PARAGRAPH_TITLE}, sentence {SENTENCE_INDEX}] {EVIDENCE}\n"
        "2. ...\n"
        "answer: ***{ANSWER}***"
    )
