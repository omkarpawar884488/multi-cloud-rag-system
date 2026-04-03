from __future__ import annotations

import re
import logging
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .settings import LLM_MODEL, TEMPERATURE, TOP_K
from .vectorstore import get_retriever


# ---------------------------
# STEP 1: Topic Profiles
# ---------------------------

@dataclass(frozen=True)
class TopicProfile:
    canonical_topic: str
    # Provider-specific keyword groups: high precision words/phrases that indicate topic
    provider_keywords: Dict[str, List[str]]
    # Optional additional query terms to improve recall for a provider that uses different vocab
    provider_query_expansions: Dict[str, List[str]]
    # Thresholds for topic gating per provider
    min_score: float = 2.0
    min_docs_per_provider: int = 1

def _contains_phrase(text: str, phrase: str) -> bool:
    # simple, deterministic phrase match (can be upgraded later)
    return phrase in text

def topic_score_doc(doc, keywords: List[str]) -> float:
    """
    Deterministic keyword scoring.
    - 1.0 per keyword/phrase hit (you can weight later)
    - uses page_content, and boosts title/url if available
    """
    content = (doc.page_content or "").lower()
    title = (doc.metadata.get("title") or "").lower()
    url = (doc.metadata.get("url") or "").lower()

    score = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        if _contains_phrase(content, kw_l):
            score += 1.0
        # small boost if present in title/url (often more precise)
        if title and _contains_phrase(title, kw_l):
            score += 0.5
        if url and _contains_phrase(url, kw_l):
            score += 0.5

    return score

# ---------------------------
# Canonical topic profiles (AWS/Azure/GCP mapping)
# ---------------------------

FOUNDATION_PROFILE = TopicProfile(
    canonical_topic="enterprise_cloud_foundation",
    provider_keywords={
        "aws": [
            "landing zone", "control tower", "aws organizations",
            "organizational unit", "service control policy", "scp",
            "multi-account", "account baseline", "guardrails", "shared services"
        ],
        "azure": [
            "landing zone", "enterprise-scale", "management group",
            "subscription", "policy assignment", "azure policy",
            "blueprint", "hub-spoke", "platform landing zone"
        ],
        "gcp": [
            # IMPORTANT: avoid requiring “landing zone” string
            "resource hierarchy", "organization", "folders", "projects",
            "organization policy", "org policy", "constraints",
            "shared vpc", "host project", "service project",
            "foundation", "blueprint", "security blueprint",
            "bootstrap", "logging project", "monitoring project", "network project"
        ],
    },
    provider_query_expansions={
        "aws": ["landing zone", "control tower", "organizations", "scp guardrails"],
        "azure": ["enterprise-scale landing zone", "management groups policy", "caf landing zone"],
        "gcp": ["resource hierarchy folders projects", "shared vpc host project", "organization policy constraints", "foundation blueprint"],
    },
    min_score=2.0,
    min_docs_per_provider=1
)

ARCHITECTURE_PROFILE = TopicProfile(
    canonical_topic="architecture_design_patterns",
    provider_keywords={
        "aws": ["reference architecture", "well-architected", "multi-tier", "event-driven", "microservices", "hub and spoke"],
        "azure": ["reference architecture", "azure architecture center", "caf", "hub-spoke", "enterprise-scale architecture"],
        "gcp": ["reference architecture", "architecture", "blueprint", "solution design", "shared vpc architecture", "project separation"],
    },
    provider_query_expansions={
        "aws": ["reference architecture", "well-architected design"],
        "azure": ["reference architecture", "caf architecture"],
        "gcp": ["reference architecture", "blueprint architecture"],
    },
    min_score=1.5,
    min_docs_per_provider=1
)

BEST_PRACTICES_PROFILE = TopicProfile(
    canonical_topic="best_practices_well_architected",
    provider_keywords={
        "aws": ["well-architected", "best practices", "design principles", "operational excellence", "reliability", "security pillar", "cost optimization"],
        "azure": ["well-architected framework", "best practices", "design principles", "cloud adoption framework", "caf"],
        "gcp": ["architecture framework", "best practices", "design principles", "reliability", "security best practices"],
    },
    provider_query_expansions={
        "aws": ["well-architected best practices"],
        "azure": ["azure well-architected framework", "caf best practices"],
        "gcp": ["google cloud architecture framework best practices"],
    },
    min_score=1.5,
    min_docs_per_provider=1
)

SERVICE_FUNDAMENTALS_PROFILE = TopicProfile(
    canonical_topic="service_fundamentals",
    provider_keywords={
        "aws": ["ec2", "vpc", "iam", "s3", "rds"],
        "azure": ["virtual machine", "vnet", "entra", "azure ad", "storage account"],
        "gcp": ["compute engine", "vpc network", "iam", "cloud storage", "projects"],
    },
    provider_query_expansions={
        "aws": [],
        "azure": [],
        "gcp": [],
    },
    min_score=1.0,
    min_docs_per_provider=1
)

SECURITY_PROFILE = TopicProfile(
    canonical_topic="security_compliance",
    provider_keywords={
        "aws": ["iam", "kms", "security hub", "guardrails", "scp", "compliance", "audit", "encryption"],
        "azure": ["rbac", "defender for cloud", "azure policy", "compliance", "audit", "encryption", "key vault"],
        "gcp": ["iam", "organization policy", "vpc service controls", "access context manager", "security command center", "compliance", "audit", "encryption"],
    },
    provider_query_expansions={
        "aws": ["security hub compliance", "kms encryption"],
        "azure": ["defender for cloud compliance", "azure policy rbac"],
        "gcp": ["vpc service controls access context manager", "security command center org policy"],
    },
    min_score=1.5,
    min_docs_per_provider=1
)

COST_PROFILE = TopicProfile(
    canonical_topic="cost_optimization_governance",
    provider_keywords={
        "aws": ["cost explorer", "budgets", "cost allocation tags", "governance", "chargeback", "showback"],
        "azure": ["cost management", "budgets", "governance", "policy", "chargeback", "showback"],
        "gcp": ["billing account", "budgets", "cost controls", "project billing", "governance", "labels"],
    },
    provider_query_expansions={
        "aws": ["cost explorer budgets tags"],
        "azure": ["azure cost management budgets"],
        "gcp": ["billing account budgets project billing labels"],
    },
    min_score=1.5,
    min_docs_per_provider=1
)

TOPIC_PROFILES: List[Tuple[List[str], TopicProfile]] = [
    (["landing zone", "foundation", "baseline", "enterprise setup", "org structure"], FOUNDATION_PROFILE),
    (["architecture", "design pattern", "reference architecture"], ARCHITECTURE_PROFILE),
    (["well-architected", "well architected", "best practices"], BEST_PRACTICES_PROFILE),
    (["security", "compliance", "iam", "audit", "encryption"], SECURITY_PROFILE),
    (["cost", "budget", "governance", "chargeback", "showback"], COST_PROFILE),
    (["what is", "explain", "fundamentals", "intro"], SERVICE_FUNDAMENTALS_PROFILE),
]

def detect_topic(question: str) -> TopicProfile:
    q = question.lower()
    for triggers, profile in TOPIC_PROFILES:
        if any(t in q for t in triggers):
            return profile
    # default safe fallback
    return ARCHITECTURE_PROFILE



logger = logging.getLogger("rag")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

def _dedupe_docs(docs):
    """Deduplicate docs by (provider + hash(page_content))."""
    seen = set()
    out = []
    for d in docs:
        prov = (d.metadata.get("provider") or "").lower()
        content = d.page_content or ""
        key = (prov, hashlib.md5(content.encode("utf-8")).hexdigest())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _log_provider_balance(question: str, docs, stage: str):
    counts = Counter([(d.metadata.get("provider") or "unknown").lower() for d in docs])
    logger.info(f"[{stage}] Q='{question[:80]}...' provider_counts={dict(counts)} total_docs={len(docs)}")
    
'''
### per_provider_retrieve: this is redundant fucntion
def per_provider_retrieve(question: str, requested: set[str], k_total: int):
    """
    Deterministic multi-cloud retrieval:
    - fetch k_per docs per requested provider
    - merge + dedupe
    - (optional) cap to k_total after merge
    """
    req = [p for p in ["aws", "azure", "gcp"] if p in requested]
    if not req:
        # No explicit provider requested -> fallback to broad
        docs = get_retriever(where=None, k=k_total).invoke(question)
        _log_provider_balance(question, docs, stage="per_provider:fallback_broad")
        return docs

    k_per = max(1, k_total // len(req))  # split budget across providers
    merged = []

    for p in req:
        docs_p = get_retriever(where={"provider": p}, k=k_per).invoke(question)
        merged.extend(docs_p)

    merged = _dedupe_docs(merged)

    # optional cap: keep best k_total docs (keep order as returned)
    merged = merged[:k_total]

    _log_provider_balance(question, merged, stage=f"per_provider:merged(k_per={k_per})")
    return merged
'''

def per_provider_retrieve_topic_aware(question: str, requested: set[str], k_total: int):
    """
    Topic-aware multi-cloud retrieval:
    - detect canonical topic
    - for each provider: fetch more candidates, then rerank by provider-specific topic keywords
    - return docs + topic_missing_providers (for scoring later)
    """
    topic = detect_topic(question)

    req = [p for p in ["aws", "azure", "gcp"] if p in requested]
    if not req:
        docs = get_retriever(where=None, k=k_total).invoke(question)
        _log_provider_balance(question, docs, stage="topicAware:fallback_broad")
        return docs, set()

    k_per = max(1, k_total // len(req))
    merged = []
    topic_missing = set()

    for p in req:
        # Stage A: recall-heavy candidates (8x per-provider budget; cap for cost)
        k_candidates = min(8 * k_per, 50)

        # Provider-specific query expansion helps mismatch cases (like GCP “landing zone”)
        expansions = topic.provider_query_expansions.get(p, [])
        query_p = question if not expansions else (question + " " + " ".join(expansions))

        candidates = get_retriever(where={"provider": p}, k=k_candidates).invoke(query_p)

        # Stage B: topic rerank
        kw = topic.provider_keywords.get(p, [])
        scored = [(d, topic_score_doc(d, kw)) for d in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep best k_per that clear topic threshold
        chosen = [d for (d, s) in scored if s >= topic.min_score][:k_per]

        if len(chosen) < topic.min_docs_per_provider:
            topic_missing.add(p)

        merged.extend(chosen)

    merged = _dedupe_docs(merged)
    merged = merged[:k_total]
    _log_provider_balance(question, merged, stage=f"topicAware:merged(topic={topic.canonical_topic}, k_per={k_per})")

    return merged, topic_missing



KNOWN_PROVIDERS = {"aws", "azure", "gcp"}

def requested_providers(question: str) -> set[str]:
    q = question.lower()
    providers = set()

    # simple but effective detection
    if re.search(r"\baws\b|amazon web services", q): providers.add("aws")
    if re.search(r"\bazure\b|microsoft azure|entra", q): providers.add("azure")
    if re.search(r"\bgcp\b|google cloud|gcp", q): providers.add("gcp")

    return providers

def providers_in_docs(docs) -> set[str]:
    p = set()
    for d in docs:
        prov = (d.metadata.get("provider") or "").lower()
        if prov in KNOWN_PROVIDERS:
            p.add(prov)
    return p

def grounding_report(question: str, docs) -> dict:
    req = requested_providers(question)
    have = providers_in_docs(docs)
    missing = req - have
    extra = have - req  # docs that are irrelevant providers (not necessarily bad)
    return {
        "requested": sorted(req),
        "present": sorted(have),
        "missing": sorted(missing),
        "extra_present": sorted(extra),
        "is_fully_grounded_for_requested": len(missing) == 0 if req else True
    }


def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "")
        url = d.metadata.get("url", "")
        parts.append(f"[{i}] {title}\n{url}\n{d.page_content}")
    return "\n\n".join(parts)

def _majority_provider(docs, threshold: float = 0.7) -> str | None:
    """Return provider name if one provider dominates docs by >= threshold."""
    providers = [d.metadata.get("provider") for d in docs if d.metadata.get("provider")]
    if not providers:
        return None

    counts = Counter(providers)
    provider, cnt = counts.most_common(1)[0]
    if cnt / len(providers) >= threshold:
        return provider
    return None

'''
## old code
def adaptive_retrieve(question: str, k: int):
    """
    - First retrieve broadly with k
    - If one provider dominates, re-retrieve filtered to that provider with k
    - If no provider dominates (multi-cloud), re-retrieve broadly with 2*k
    """
    # Stage 1: broad retrieve with base k
    docs = get_retriever(where=None, k=k).invoke(question)

    dominant = _majority_provider(docs, threshold=0.7)

    # Stage 2a: single-provider dominance -> filter and keep k
    if dominant:
        return get_retriever(where={"provider": dominant}, k=k).invoke(question)

    # Stage 2b: multi-cloud/ambiguous -> increase recall
    # return get_retriever(where=None, k=2 * k).invoke(question)  ## if the number of tokens increases drastically.
    return get_retriever(where=None, k=min(2 * k, 20)).invoke(question)



def adaptive_retrieve(question: str, k: int):
    # returns (docs, diag)
    """
    Retrieval policy:
    - If question explicitly asks multiple providers -> per-provider retrieval (balanced)
    - Else -> existing behavior (broad -> dominant -> filter OR broaden)
    """
    req = requested_providers(question)

    # MULTI-CLOUD: deterministic balanced retrieval
    if len(req) >= 2:
        topic = detect_topic(question)
        docs, topic_missing = per_provider_retrieve_topic_aware(
            question, requested=req, k_total=min(2 * k, 20)
        )
        diag = {
            "mode": "multicloud_topic_aware",
            "canonical_topic": topic.canonical_topic,
            "requested_providers": sorted(req),
            "topic_missing_providers": sorted(topic_missing),
        }
        return docs, diag

    # SINGLE/NO-PROVIDER: keep Option D
    docs = get_retriever(where=None, k=k).invoke(question)
    _log_provider_balance(question, docs, stage="optionD:stage1_broad")

    dominant = _majority_provider(docs, threshold=0.7)

    if dominant:
        docs2 = get_retriever(where={"provider": dominant}, k=k).invoke(question)
        _log_provider_balance(question, docs2, stage=f"optionD:stage2_filtered({dominant})")
        return docs2

    docs2 = get_retriever(where=None, k=min(2 * k, 20)).invoke(question)
    _log_provider_balance(question, docs2, stage="optionD:stage2_broadened")
    return docs2

'''

def adaptive_retrieve(question: str, k: int):
    """
    Returns: (docs, diag)
    Retrieval policy:
    - If question explicitly asks multiple providers -> per-provider retrieval (balanced + topic aware)
    - Else -> existing behavior (broad -> dominant -> filter OR broaden)
    """
    req = requested_providers(question)

    # MULTI-CLOUD: deterministic balanced retrieval
    if len(req) >= 2:
        topic = detect_topic(question)
        docs, topic_missing = per_provider_retrieve_topic_aware(
            question, requested=req, k_total=min(2 * k, 20)
        )
        diag = {
            "mode": "multicloud_topic_aware",
            "canonical_topic": topic.canonical_topic,
            "requested_providers": sorted(req),
            "topic_missing_providers": sorted(topic_missing),
        }
        return docs, diag

    # SINGLE/NO-PROVIDER: keep Option D
    topic = detect_topic(question)  # <-- add this (for consistent diag)
    docs = get_retriever(where=None, k=k).invoke(question)
    _log_provider_balance(question, docs, stage="optionD:stage1_broad")

    dominant = _majority_provider(docs, threshold=0.7)

    if dominant:
        docs2 = get_retriever(where={"provider": dominant}, k=k).invoke(question)
        _log_provider_balance(question, docs2, stage=f"optionD:stage2_filtered({dominant})")
        diag = {
            "mode": "single_or_unspecified",
            "canonical_topic": topic.canonical_topic,
            "requested_providers": sorted(req),
            "dominant_provider": dominant,
            "topic_missing_providers": [],  # not applicable here
        }
        return docs2, diag

    docs2 = get_retriever(where=None, k=min(2 * k, 20)).invoke(question)
    _log_provider_balance(question, docs2, stage="optionD:stage2_broadened")
    diag = {
        "mode": "single_or_unspecified",
        "canonical_topic": topic.canonical_topic,
        "requested_providers": sorted(req),
        "dominant_provider": None,
        "topic_missing_providers": [],
    }
    return docs2, diag


def topic_coverage_report(question: str, docs, topic: TopicProfile, requested: set[str]) -> dict:
    """
    Topic coverage per provider, based on topic_score_doc thresholding.
    This detects: provider present, but topic-evidence missing.
    """
    present_topic = set()
    provider_best = {}

    for p in requested:
        kw = topic.provider_keywords.get(p, [])
        scores = [
            topic_score_doc(d, kw)
            for d in docs
            if (d.metadata.get("provider") or "").lower() == p
        ]
        best = max(scores) if scores else 0.0
        provider_best[p] = best

        if best >= topic.min_score:
            present_topic.add(p)

    missing_topic = requested - present_topic

    return {
        "canonical_topic": topic.canonical_topic,
        "requested": sorted(requested),
        "topic_present": sorted(present_topic),
        "topic_missing": sorted(missing_topic),
        "provider_best_topic_score": provider_best,
        "is_topic_complete_for_requested": len(missing_topic) == 0 if requested else True,
    }


# ---------------------------
# Semantic grounding: "is the content actually about the asked concept?"
# ---------------------------

# Minimal "must-have" signals per provider for some common subtopics.
# Keep it deterministic: keyword presence is enough for now.
SEMANTIC_SIGNALS = {
    "iam_fundamentals": {
        "aws": ["iam", "policy", "policies", "role", "roles", "user", "users", "group", "groups", "sts", "assume role", "permission boundary"],
        "azure": ["entra", "azure ad", "microsoft entra", "rbac", "role assignment", "managed identity", "managed identities", "tenant"],
        "gcp": ["iam", "binding", "bindings", "principal", "principals", "service account", "service accounts", "roles/", "predefined roles", "custom role"],
    },
    "landing_zone_foundation": {
        "aws": ["control tower", "aws organizations", "organizational unit", "service control policy", "scp", "multi-account", "guardrails", "account baseline"],
        "azure": ["enterprise-scale", "management group", "subscription", "azure policy", "landing zone", "platform landing zone"],
        "gcp": ["resource hierarchy", "folders", "projects", "organization policy", "org policy", "shared vpc", "host project", "service project", "blueprint"],
    },
}

def _pick_semantic_subtopic(question: str, topic: TopicProfile) -> str:
    """
    Pick a semantic subtopic for signal checking.
    You can expand this mapping later.
    """
    q = question.lower()

    # IAM questions often get classified as security_compliance today; we want IAM-specific signals.
    if "iam" in q and any(t in q for t in ["explain", "what is", "difference", "differences", "compare"]):
        return "iam_fundamentals"

    if topic.canonical_topic == "enterprise_cloud_foundation":
        return "landing_zone_foundation"

    # default fallback: use topic name as a hint (no strong semantic check)
    return "none"


def semantic_grounding_report(question: str, docs, topic: TopicProfile, requested: set[str]) -> dict:
    """
    Check: for each requested provider, do retrieved docs contain semantic signals
    for the intended subtopic (e.g., IAM fundamentals), not just provider presence.
    """
    subtopic = _pick_semantic_subtopic(question, topic)

    if subtopic == "none" or subtopic not in SEMANTIC_SIGNALS:
        return {
            "subtopic": subtopic,
            "requested": sorted(requested),
            "providers_semantic_ok": sorted(requested),
            "providers_semantic_missing": [],
            "provider_signal_hits": {p: [] for p in requested},
            "is_semantically_grounded": True if requested else True,
        }

    signals = SEMANTIC_SIGNALS[subtopic]
    provider_hits = {}
    missing = set()

    # Build a provider -> concatenated text blob of retrieved docs
    for p in requested:
        text = " ".join(
            (d.page_content or "").lower()
            for d in docs
            if (d.metadata.get("provider") or "").lower() == p
        )
        kw = signals.get(p, [])
        hits = [k for k in kw if k in text]
        provider_hits[p] = hits

        # Require at least N hits to call it semantically grounded
        # Tune N later; start with 2 to reduce false positives.
        if len(hits) < 2:
            missing.add(p)

    ok = requested - missing

    return {
        "subtopic": subtopic,
        "requested": sorted(requested),
        "providers_semantic_ok": sorted(ok),
        "providers_semantic_missing": sorted(missing),
        "provider_signal_hits": provider_hits,
        "is_semantically_grounded": len(missing) == 0 if requested else True,
    }


def auto_retry_retrieve(question: str, k: int, max_retries: int = 1):
    """
    Wrap adaptive_retrieve with semantic validation.
    If semantic grounding fails for some requested providers, re-retrieve for those providers
    using targeted query expansions and merge results.
    Returns: (docs, diag, semantic_report)
    """
    docs, diag = adaptive_retrieve(question, k=k)

    req = requested_providers(question)
    topic = detect_topic(question)

    sem = semantic_grounding_report(question, docs, topic=topic, requested=req)

    if sem["is_semantically_grounded"] or not req or max_retries <= 0:
        return docs, diag, sem

    # Retry only for missing providers with subtopic-specific expansions
    subtopic = sem["subtopic"]
    missing_providers = sem["providers_semantic_missing"]

    # Build expansions: prefer our semantic signals -> use them as query terms
    expansions_map = SEMANTIC_SIGNALS.get(subtopic, {})

    merged = list(docs)

    # Give extra budget only to missing providers
    k_retry_per_provider = max(2, min(10, k // max(1, len(missing_providers))))

    for p in missing_providers:
        extra_terms = expansions_map.get(p, [])
        # Keep query short-ish: pick top few terms
        extra_terms = extra_terms[:6]
        query_p = question + " " + " ".join(extra_terms)

        retry_docs = get_retriever(where={"provider": p}, k=k_retry_per_provider).invoke(query_p)
        merged.extend(retry_docs)

    merged = _dedupe_docs(merged)
    merged = merged[:min(2 * k, 25)]  # cap to avoid huge context

    # Recompute semantic report after retry
    sem2 = semantic_grounding_report(question, merged, topic=topic, requested=req)

    diag2 = {
        **diag,
        "auto_retry": True,
        "auto_retry_subtopic": subtopic,
        "auto_retry_missing_providers": missing_providers,
        "auto_retry_k_per_provider": k_retry_per_provider,
        "semantic_grounding_before": sem,
        "semantic_grounding_after": sem2,
    }

    return merged, diag2, sem2




def build_chain(where: dict | None = None):
    """
    Builds the RAG pipeline:
    - retrieval (adaptive + topic-aware)
    - deterministic reranking
    - semantic grounding validation
    - LLM answer generation
    """
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a cloud platform assistant (AWS/Azure/GCP).\n"
    
     "Use ONLY the provided context to answer, but explain clearly and completely.\n"
    
     "Your answer must:\n"
     "1. Be correct and based on the context\n"
     "2. Be well-structured and easy to understand\n"
     "3. Cover all important aspects of the question\n"
    
     "If multiple providers (AWS/Azure/GCP) are requested:\n"
     "- You MUST compare them explicitly\n"
     "- Mention each provider separately\n"
     "- Highlight key differences\n"
    
     "If context for a provider is missing:\n"
     "- Clearly say it is missing\n"
     "- Do NOT hallucinate\n"
    
     "Always cite sources like [1], [2]."
    ),
    ("human",
     "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations:")
    ])

    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    parser = StrOutputParser()

    # If caller explicitly sets where, respect it. Otherwise do adaptive retrieval (Option D).
    '''
    ## old code
    def get_docs(q: str):
        if where:
            return get_retriever(where=where, k=TOP_K).invoke(q)
        return adaptive_retrieve(q, k=TOP_K)
    

    def get_docs(q: str):
        if where:
            docs = get_retriever(where=where, k=TOP_K).invoke(q)
            return docs, {"mode": "forced_where", "canonical_topic": detect_topic(q).canonical_topic, "requested_providers": sorted(requested_providers(q))}
        return adaptive_retrieve(q, k=TOP_K)
     '''
    def get_docs(q: str):
        if where:
            docs = get_retriever(where=where, k=TOP_K).invoke(q)
            topic = detect_topic(q)
            req = requested_providers(q)
            sem = semantic_grounding_report(q, docs, topic=topic, requested=req)
            return docs, {
                "mode": "forced_where",
                "canonical_topic": topic.canonical_topic,
                "requested_providers": sorted(req),
            }, sem
    
        docs, diag, sem = auto_retry_retrieve(q, k=TOP_K, max_retries=1)
        return docs, diag, sem
    
       

    def enforce_grounding(question: str, docs):
        rep = grounding_report(question, docs)
        if rep["requested"] and rep["missing"]:
            # Hard block: return a safe refusal-like answer context
            # so the LLM can only respond "missing context" (or you can return directly)
            missing_str = ", ".join(rep["missing"])
            return docs, (
                f"NOTE: Missing retrieved context for provider(s): {missing_str}. "
                f"You must not answer those parts."
            )
        return docs, ""

    chain = (
        {"question": RunnablePassthrough()}
    
        # 1) Retrieve docs + diagnostics
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "docs_diag": get_docs(x["question"]),  # (docs, diag)
        })
    
        # 2) Enforce grounding note using docs_diag[0] (docs)
        | RunnableLambda(lambda x: {
            **x,
            "docs": x["docs_diag"][0],
            "retrieval_diag": x["docs_diag"][1],
            "semantic_diag": x["docs_diag"][2],
        })
        | RunnableLambda(lambda x: {
            **x,
            "grounding_note": enforce_grounding(x["question"], x["docs"])[1],
        })
    
        # 3) Build context
        | RunnableLambda(lambda x: {
            **x,
            "context": x["grounding_note"] + "\n\n" + format_docs(x["docs"]),
        })
    
        # 4) Call LLM to produce answer
        | RunnableLambda(lambda x: {
            **x,
            "answer": (prompt | llm | parser).invoke({
                "question": x["question"],
                "context": x["context"],
            })
        })
    
        # 5) Attach reports + sources
        | RunnableLambda(lambda x: {
            "answer": x["answer"],
            "grounding": grounding_report(x["question"], x["docs"]),
            "topic_coverage": topic_coverage_report(
                x["question"],
                x["docs"],
                detect_topic(x["question"]),
                requested_providers(x["question"])
            ),
            "retrieval": x["retrieval_diag"],
            "semantic_grounding": x["semantic_diag"],
            # this is new line
            "docs": x["docs"],
            
            "sources": [
                {
                    "rank": i + 1,
                    "title": d.metadata.get("title"),
                    "url": d.metadata.get("url"),
                    "provider": d.metadata.get("provider"),
                    "category": d.metadata.get("category"),
                }
                for i, d in enumerate(x["docs"])
            ],
        })
    )

    return chain





if __name__ == "__main__":
    where = {"provider": "aws", "category": "well_architected"}
    chain = build_chain(where=where)

    q = "Explain cost optimization pillar in simple terms and give 3 actionable checks."
    out = chain.invoke(q)

    print("\nANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(s)
