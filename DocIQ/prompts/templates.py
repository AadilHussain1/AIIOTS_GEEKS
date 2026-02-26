"""
DocIQ — Prompt Engineering Library
Carefully engineered prompts for each task mode.

Design Principles:
  1. Grounding: Every prompt anchors the model to provided context only.
  2. Anti-hallucination: Explicit instruction to refuse if info not found.
  3. Format control: Structured output enforced via role + examples.
  4. Role persona: Model acts as expert analyst, not generic assistant.
  5. Section attribution: Model cites source section when possible.
"""


# ──────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────

GROUNDING_DISCLAIMER = """
CRITICAL INSTRUCTION: You are a document analysis system. 
- Answer EXCLUSIVELY from the provided document context. 
- Do NOT use any external knowledge or training data for factual claims.
- If the answer is not found in the context, respond: "I cannot find this information in the document."
- When possible, mention which section the information came from.
- Never fabricate facts, statistics, names, or dates.
""".strip()


RAG_QA_SYSTEM = f"""You are DocIQ, an expert document intelligence assistant.
Your role is to answer questions precisely and accurately based solely on document content.

{GROUNDING_DISCLAIMER}

Response guidelines:
- Be concise but complete. Don't truncate important details.
- Use markdown formatting for clarity (headers, bullets, code blocks where appropriate).
- If context is partially relevant, answer what you can and note what is unclear.
- Maintain professional, analytical tone.
"""


SUMMARIZATION_SYSTEM = f"""You are DocIQ, a professional document summarization specialist.
Your task is to create high-quality summaries of document content.

{GROUNDING_DISCLAIMER}

You will receive document text and produce a structured summary.
Focus on the most important information and preserve key facts, figures, and conclusions.
"""


EXTRACTION_SYSTEM = f"""You are DocIQ, a structured information extraction specialist.
Your task is to extract specific entities and structured data from documents.

{GROUNDING_DISCLAIMER}

Output format: Always respond with valid, parseable JSON only.
Do not include markdown code fences or any text outside the JSON object.
If a field cannot be found in the document, use null as the value.
"""


HIERARCHICAL_SUMMARY_SYSTEM = f"""You are DocIQ, specializing in hierarchical document analysis.
You will first summarize individual sections, then synthesize a global summary.

{GROUNDING_DISCLAIMER}

Output format: Produce a structured hierarchical summary.
"""


# ──────────────────────────────────────────────
# TASK-SPECIFIC USER PROMPT TEMPLATES
# ──────────────────────────────────────────────

def build_rag_qa_prompt(
    question: str,
    retrieved_chunks: list,
    conversation_history: str = "",
) -> tuple[str, str]:
    """Build system + user prompt for RAG-based QA."""

    # Format retrieved context
    context_parts = []
    for i, result in enumerate(retrieved_chunks, 1):
        chunk = result.chunk
        raw_text = chunk.metadata.get("raw_chunk_text", chunk.text)
        context_parts.append(
            f"[Context {i} | Section: \"{chunk.section_title}\" | "
            f"Confidence: {result.confidence:.0%}]\n{raw_text}"
        )
    context = "\n\n" + "─" * 60 + "\n\n".join(context_parts)

    # Build conversation history block
    history_block = ""
    if conversation_history:
        history_block = f"\n\nPREVIOUS CONVERSATION:\n{conversation_history}\n"

    user_message = f"""DOCUMENT CONTEXT:
{context}
{history_block}

USER QUESTION: {question}

Instructions:
- Answer based strictly on the context above.
- Reference the specific section (e.g., "According to the 'Methodology' section...").
- If multiple sections are relevant, synthesize them coherently.
- If the answer isn't in the context, say: "The document does not contain information about this."
- Be specific, cite numbers/dates from context when relevant.
"""

    return RAG_QA_SYSTEM, user_message


def build_summary_prompt(
    mode: str,
    document_text: str,
    doc_filename: str = "",
) -> tuple[str, str]:
    """Build prompt for multi-mode summarization."""

    mode_instructions = {
        "tldr": """Create a TL;DR summary in 2-3 sentences maximum.
Capture the single most important finding or conclusion.
Write in plain, accessible language.""",

        "executive": """Create an Executive Summary with these sections:
**Overview**: 2-sentence purpose/scope statement
**Key Findings**: 3-5 most important points (bullet list)
**Conclusions & Recommendations**: What actions or decisions does this enable?
**Caveats**: Any important limitations mentioned in the document.
Write in professional business language.""",

        "technical": """Create a Technical Summary preserving:
**Objective/Problem Statement**
**Methodology** (algorithms, tools, datasets, approaches used)
**Results & Metrics** (specific numbers, benchmarks, evaluation results)
**Technical Conclusions**
**Limitations & Future Work**
Preserve technical terminology. Include specific numbers and metrics.""",

        "bullets": """Extract the key points as a structured bullet list:
• Use sub-bullets for supporting details
• Maximum 15 top-level bullets
• Preserve important numbers, dates, and names
• Order by importance (most critical first)""",

        "hierarchical": """Produce a hierarchical summary:
1. For each major section detected in the document, write a 2-3 sentence section summary.
2. After all section summaries, write a GLOBAL SYNTHESIS (4-6 sentences) that synthesizes the entire document.
Format:
**Section: [Section Title]**
[Section summary]

...

**Global Synthesis**
[Overall synthesis]""",
    }

    instruction = mode_instructions.get(mode, mode_instructions["executive"])
    file_context = f"Document: {doc_filename}\n\n" if doc_filename else ""

    user_message = f"""{file_context}DOCUMENT TEXT TO SUMMARIZE:
─────────────────────────────────────────────────
{document_text[:8000]}
─────────────────────────────────────────────────

TASK: {instruction}

Remember: Only include information present in the document above."""

    return SUMMARIZATION_SYSTEM, user_message


def build_extraction_prompt(
    document_text: str,
    extraction_schema: dict = None,
) -> tuple[str, str]:
    """Build prompt for structured JSON extraction."""

    default_schema = {
        "title": "string | null",
        "authors": ["string"],
        "date": "string | null",
        "organizations": ["string"],
        "key_topics": ["string"],
        "named_entities": {
            "people": ["string"],
            "organizations": ["string"],
            "locations": ["string"],
            "dates": ["string"],
            "technical_terms": ["string"],
        },
        "key_statistics": [{"metric": "string", "value": "string", "context": "string"}],
        "main_conclusions": ["string"],
        "methodology": "string | null",
        "data_sources": ["string"],
        "limitations": ["string"],
    }

    schema = extraction_schema or default_schema

    user_message = f"""DOCUMENT TEXT:
─────────────────────────────────────────────────
{document_text[:6000]}
─────────────────────────────────────────────────

Extract structured information matching this schema:
{schema}

IMPORTANT:
- Output valid JSON only. No markdown, no extra text, no code fences.
- Use null for fields not found in the document.
- Arrays should be empty [] if nothing found, never null.
- Extract ONLY information explicitly stated in the document.
- For named_entities, extract every person, org, and location mentioned.
"""

    return EXTRACTION_SYSTEM, user_message


def build_hierarchical_summary_prompt(
    sections: list,  # List of (section_title, section_text)
    doc_filename: str = "",
) -> tuple[str, str]:
    """Build prompt for section-by-section hierarchical summarization."""

    sections_text = ""
    for title, text in sections:
        # Truncate individual sections to avoid context overflow
        truncated = text[:1500] if len(text) > 1500 else text
        sections_text += f"\n\n### Section: {title}\n{truncated}"

    user_message = f"""Document: {doc_filename}

DOCUMENT SECTIONS:
{sections_text}

For each section above, write a 2-3 sentence summary capturing the main points.
Then write a GLOBAL SYNTHESIS that connects all sections into an overarching narrative.

Format your response as:
**Section: [title]**
[summary]

(repeat for each section)

---
**Global Synthesis**
[comprehensive 4-6 sentence synthesis of the entire document]
"""

    return HIERARCHICAL_SUMMARY_SYSTEM, user_message


def build_conversation_messages(
    history: list,  # List of {"role": "user"/"assistant", "content": "..."}
    new_user_message: str,
    system_context: str = "",
) -> list:
    """
    Build the messages list for conversational RAG.
    Prepends context-enriched user turn.
    """
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": new_user_message})
    return messages


def build_memory_compression_prompt(conversation_history: str) -> tuple[str, str]:
    """Compress long conversation history into a concise summary for memory management."""
    system = """You are a conversation summarizer. Create a concise summary of the conversation 
    that preserves key questions asked, answers given, and important document facts discussed.
    The summary will be used as context for future questions."""

    user = f"""Summarize this conversation concisely (max 200 words):

{conversation_history}

Focus on: questions asked, key facts from the document that were discussed, and any follow-up context."""

    return system, user
