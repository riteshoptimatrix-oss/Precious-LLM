# PRECIOUS AI — COMPLETE WEBSITE DATA PERFECTION & KNOWLEDGE BASE RECONSTRUCTION

You are working on the existing Precious AI project for Precious Education.

Your task is to make the ENTIRE existing website knowledge/data layer extremely clean, accurate, complete, consistent, structured, searchable, and production-ready.

This is NOT only an IELTS fix.

You must audit and fix ALL available website/business data contained in the current `data/` directory and any existing website knowledge JSON files.

You are explicitly allowed to MODIFY, REWRITE, MERGE, SPLIT, NORMALIZE, RESTRUCTURE, or REGENERATE the existing JSON data files when required.

DO NOT blindly preserve bad existing data.

DO NOT destroy useful information.

DO NOT invent information.

DO NOT use external AI APIs.

DO NOT use OpenAI, Gemini, Claude, Groq, Anthropic, OpenRouter, external embeddings, vector databases, or pretrained models.

The project must continue using the existing custom LLM and local knowledge/retrieval architecture.

---

# 1. PRIMARY OBJECTIVE

The final website knowledge dataset must be good enough that Precious AI can answer natural client questions accurately using the actual available Precious Education information.

The system must not behave like this:

User:
"Can you give me the details of IELTS class?"

Bad:
"Yes! We provide high-quality IELTS training with expert instructors..."

when the dataset actually contains much more detailed IELTS information.

Instead, the dataset must preserve and expose ALL relevant facts so retrieval and the custom LLM can construct a complete answer.

The same principle applies to:

* IELTS
* Study Visa
* Immigration
* Work Visa
* Visitor Visa
* Inadmissibility
* Admissions
* Countries
* Contact
* About Precious Education
* Other services
* Any other information present in the source data

---

# 2. FIRST: FULL DATA AUDIT

Before changing anything, recursively inspect the entire:

`data/`

directory.

Find every:

* `.json`
* `.jsonl`
* related structured data file
* website scrape output
* page data
* chunk data
* metadata file
* category file
* country file
* service file

Create an inventory.

For every file determine:

* filename
* schema
* number of records
* source
* content type
* duplicate records
* missing fields
* malformed records
* empty records
* irrelevant records
* conflicting records
* useful information
* source URL
* page title
* page/section/category information

DO NOT modify files until the audit is complete.

---

# 3. PRESERVE RAW DATA

Never permanently destroy the original data before creating a backup.

Create:

`data/raw_backup/`

or an equivalent safe backup location.

Preserve the original files exactly as received.

The cleaned/reconstructed dataset must be separate from the raw source.

This is required for rollback and auditing.

---

# 4. DO NOT INVENT DATA

This is one of the most important rules.

You may:

* clean text
* fix formatting
* remove duplication
* normalize capitalization
* normalize whitespace
* merge duplicate records
* split unrelated content
* reorganize categories
* improve metadata
* create structured fields from explicit source content
* preserve synonyms
* create search keywords from existing content

You MUST NOT invent:

* visa requirements
* fees
* processing times
* success rates
* guarantees
* eligibility requirements
* company ownership
* founder information
* services
* class timings
* course duration
* IELTS scores
* admission requirements
* government policies
* legal advice
* immigration outcomes

unless the source data explicitly contains those facts.

If information is unavailable:

`not_available`

or an equivalent explicit state must be used.

Never fill missing information with guesses.

---

# 5. CREATE A CANONICAL DATA MODEL

Do not allow every JSON file to have a completely different schema.

Create a normalized canonical knowledge structure.

Each knowledge record should support fields such as:

```json
{
  "knowledge_id": "",
  "knowledge_type": "",
  "category": "",
  "subcategory": "",
  "country": "",
  "service": "",
  "topic": "",
  "title": "",
  "content": "",
  "facts": [],
  "keywords": [],
  "synonyms": [],
  "source_url": "",
  "source_file": "",
  "source_page": "",
  "source_section": "",
  "source_title": "",
  "content_hash": "",
  "record_version": "",
  "active": true
}
```

Do not force irrelevant fields to contain fake values.

Use:

`null`

or:

`not_available`

where appropriate.

---

# 6. INFORMATION HIERARCHY

Organize the data logically.

Recommended hierarchy:

Precious Education
│
├── About
│
├── Services
│   ├── Immigration Visa Services
│   ├── Work Visa Services
│   ├── Visit Visa Services
│   ├── Inadmissibility Services
│   ├── Study Visa Services
│   ├── Other Services
│   └── IELTS Coaching
│
├── Countries
│   ├── Australia
│   ├── Canada
│   ├── New Zealand
│   ├── UK
│   ├── USA
│   ├── Singapore
│   ├── Malaysia
│   └── Ireland
│
├── Contact
│
└── Other Website Information

Use only categories actually supported by the source data.

---

# 7. IELTS DATA — COMPLETE RECONSTRUCTION

IELTS must receive special attention because it is currently producing incomplete answers.

Find EVERY IELTS-related record across ALL JSON files.

Search for:

* IELTS
* IELTS coaching
* IELTS class
* IELTS classes
* IELTS training
* IELTS preparation
* British Council
* IDP
* IELTS registration
* faculty
* coaching
* preparatory materials
* study materials
* CDs
* library
* brochures
* video tapes
* registration center

Do not assume IELTS information exists in only one file.

Merge all relevant IELTS information from the entire dataset.

Remove duplicate fragments.

Preserve every unique factual detail.

Create clean IELTS records such as:

```json
{
  "knowledge_type": "service",
  "category": "ielts",
  "topic": "ielts_coaching",
  "title": "IELTS Coaching",
  "content": "...",
  "facts": [
    "...",
    "...",
    "..."
  ],
  "keywords": [
    "IELTS",
    "IELTS coaching",
    "IELTS classes",
    "IELTS training",
    "IELTS preparation"
  ]
}
```

If the source explicitly contains details about:

* experienced faculty
* British Council-trained faculty
* personalized coaching
* preparatory materials
* CDs
* library
* brochures
* video tapes
* British Council registration
* IDP registration
* authorized registration center

all of those facts must remain available in the cleaned dataset.

Do not reduce them to:

"IELTS coaching is provided."

---

# 8. SERVICE DATA

Audit every service.

For each service identify:

* service name
* category
* subcategory
* countries, if explicitly associated
* description
* available options
* process information
* supporting information
* source URL
* source section

Examples include:

Immigration Visa Services:

* Express Entry
* Family Sponsorship
* OINP
* PNP
* RCIP
* AIPP
* Agri-Food Pilot
* PR Card Renewal
* Citizenship

Work Visa Services:

* LMIA
* Employer Specific Work Permit
* renewals/extensions
* PGWP
* Open Work Permits
* Bridging Open Work Permit
* Spousal Open Work Permit
* IEC
* Status Restoration

Visit:

* TRV
* Visitor Visa
* Visitor Record Extension
* Super Visa

Inadmissibility:

* Temporary Residence Permit
* Misrepresentation
* Procedural Fairness Letter
* H&C considerations
* Medical Inadmissibility
* Residency Obligation

Study:

* Study Permit / Student Visa
* Study Permit Extension
* College/University Admission
* College Transfer / DLI Change

Other:

* ATIP Notes
* IELTS Coaching
* Travel/Super Visa Insurance
* Pre/Post Landing Services
* amended temporary resident documents

ONLY include items actually supported by source data.

---

# 9. COUNTRY DATA

Normalize country-specific information.

Each country should have its own clean knowledge records.

For example:

USA

should not accidentally retrieve:

Canada SDS
Canada PNP
Canada LMIA

unless the user asks about Canada.

Similarly:

Canada

should not be mixed with:

USA F1/M1
Australia subclass information
UK student visa

unless explicitly relevant.

Use:

`country`

as a first-class retrieval field.

---

# 10. VISA CATEGORY NORMALIZATION

Normalize common terminology without changing meaning.

Examples:

"student visa"
"study visa"
"study permit"

may be related terms, but DO NOT automatically treat them as identical legal concepts.

Preserve the original factual meaning.

Examples of searchable terms:

USA:

* F1
* M1
* F2
* M2

Canada:

* Study Permit
* Student Visa
* Visitor Visa
* Work Permit
* LMIA
* PGWP
* PNP

Australia:

* Student Visa
* subclass 500

etc.

Only include terminology supported by source data.

---

# 11. CONTACT DATA

Create dedicated authoritative contact records.

Include every explicitly available:

* email
* phone
* address
* office
* contact page
* official website

For example, if the source contains:

`info@preciousedu.in`

it must be stored as a dedicated contact fact.

Do not make the LLM search through a generic service paragraph to answer an email query.

Likewise for:

* phone number
* address
* office location

---

# 12. COMPANY IDENTITY

Create dedicated records for:

* company name
* organization description
* establishment information
* business type
* services
* countries
* other explicitly available company information

For:

"Who are you?"

the system should retrieve company identity information.

For:

"Who is the owner?"

only return owner information if explicitly present.

NEVER infer ownership from:

* director names
* founders guessed from text
* employees
* domain ownership
* search engine results

---

# 13. SOURCE PROVENANCE

Every normalized record must retain provenance.

At minimum:

* source file
* source URL
* source page
* source section
* source title
* source content hash
* ingestion version

This is mandatory.

If a client asks something, the system should be able to determine where the answer came from.

---

# 14. REMOVE DUPLICATES

Detect duplicate records using:

* exact text
* normalized text
* content hash
* URL + section
* semantic-like deterministic keyword overlap

Do not use external embeddings.

If two records contain exactly the same information:

merge them.

If two records partially overlap:

merge only the duplicated facts while preserving unique facts.

Do not accidentally remove unique information.

---

# 15. REMOVE CORRUPTED CONTENT

Remove or repair:

* HTML garbage
* JavaScript
* CSS
* navigation menus
* cookie banners
* tracking text
* repeated footer text
* malformed Unicode
* excessive whitespace
* duplicated headings
* broken fragments
* empty records
* meaningless metadata

Preserve meaningful text.

---

# 16. FIX BROKEN CONTENT

Correct only obvious structural/formatting issues.

Examples:

Bad:

`IELTS    Coaching    Classes`

Normalize to:

`IELTS Coaching Classes`

Bad:

`Study Visa\n\n\n\nCanada`

Normalize appropriately.

Do NOT rewrite factual meaning.

---

# 17. CREATE FACT-LEVEL DATA

Where practical, extract explicit factual statements into a `facts` array.

Example:

```json
{
  "topic": "ielts_coaching",
  "facts": [
    "Precious Education provides IELTS coaching.",
    "The faculty includes experienced and British Council-trained faculty members.",
    "Personalized coaching is provided.",
    "Preparatory materials are available.",
    "The resources include CDs.",
    "A library is available with brochures and video tapes.",
    "The organization is an authorized center to accept British Council and IDP IELTS registrations."
  ]
}
```

IMPORTANT:

These facts must come ONLY from the source material.

Do not create new claims.

---

# 18. CREATE SEARCH METADATA

For every record generate deterministic:

* keywords
* synonyms
* normalized terms
* aliases
* country terms
* service terms
* topic terms

Example IELTS:

```json
"keywords": [
  "IELTS",
  "IELTS class",
  "IELTS classes",
  "IELTS coaching",
  "IELTS training",
  "IELTS preparation",
  "IELTS registration",
  "British Council",
  "IDP"
]
```

Do not add unrelated keywords merely to increase retrieval.

---

# 19. CHUNKING

If content is too large, split it into logical chunks.

Do NOT split in the middle of a factual concept.

Prefer:

heading
→ subsection
→ related paragraphs
→ related facts

Each chunk must retain:

* knowledge_id
* chunk_id
* category
* country
* service
* topic
* title
* section
* content
* source URL
* source metadata

For broad informational questions, multiple chunks must be retrievable.

---

# 20. RETRIEVAL REQUIREMENTS

After fixing the data, inspect the existing website retrieval code.

It must support:

* exact matching
* keyword matching
* phrase matching
* title matching
* heading matching
* topic matching
* service matching
* country matching
* synonym matching
* multi-chunk retrieval

For:

"Can you give me the details of IELTS class?"

retrieval must be capable of returning all relevant IELTS chunks.

It must not return only one generic sentence.

---

# 21. RELEVANCE SCORING

Implement or improve deterministic ranking.

Suggested:

Exact phrase: +15
Exact topic: +12
Exact service: +12
Exact country: +12
Keyword match: +8
Title match: +8
Heading match: +7
Section match: +6
Synonym match: +3
Multiple keyword overlap: +5

Penalties:

Wrong country: -15
Wrong service: -15
Wrong visa category: -15
Unrelated category: -10
Very weak overlap: -8

The ranking system must prevent common/high-frequency Canada content from appearing for USA questions.

---

# 22. KNOWLEDGE ROUTING

Ensure the router distinguishes:

`NO_KNOWLEDGE_REQUIRED`

`WEBSITE_KNOWLEDGE_REQUIRED`

`PROJECT_KNOWLEDGE_REQUIRED`

`WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED`

Examples:

"Hello"

→ NO_KNOWLEDGE_REQUIRED

"Who are you?"

→ WEBSITE_KNOWLEDGE_REQUIRED

"What is your email?"

→ WEBSITE_KNOWLEDGE_REQUIRED

"Tell me about IELTS coaching."

→ WEBSITE_KNOWLEDGE_REQUIRED

"What is the status of project ABC?"

→ PROJECT_KNOWLEDGE_REQUIRED

"Tell me about project ABC and your immigration services."

→ WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED

---

# 23. ANSWER GENERATION

The custom LLM must receive the selected relevant knowledge.

The prompt must instruct it:

* answer the user's actual question
* use all relevant retrieved facts
* do not ignore useful retrieved facts
* do not invent missing facts
* do not mix countries
* do not mix unrelated services
* be concise but sufficiently detailed
* use bullets for multiple details
* answer naturally
* do not blindly copy the database
* do not use generic filler
* do not redirect to the website instead of answering

---

# 24. "DETAILS" QUESTIONS

For queries containing concepts such as:

* details
* tell me about
* explain
* information
* everything about
* what do you provide
* what facilities
* what services
* describe

retrieve multiple relevant records.

The final answer should cover the major available facts.

Do NOT give a one-line answer if detailed source information exists.

---

# 25. FOLLOW-UP QUESTIONS

Preserve conversation context.

Example:

User:
"Tell me about IELTS coaching."

Assistant:
IELTS details.

User:
"How does it work?"

The system should understand that "it" refers to IELTS coaching.

Do not repeat unrelated information.

Do not lose the previous topic.

---

# 26. UNKNOWN INFORMATION

If the source data does not contain the requested information:

DO NOT hallucinate.

Example:

"What is your IELTS fee?"

If fee is not in the source:

"I don't currently have the IELTS fee information in the available Precious Education data."

Do NOT invent a fee.

Same for:

* timings
* duration
* guarantees
* eligibility
* processing time
* requirements

unless explicitly available.

---

# 27. DATA CONSISTENCY CHECK

After reconstruction, run automated consistency checks.

Check for:

* same service with different names
* contradictory descriptions
* duplicate facts
* wrong country labels
* wrong service labels
* missing source metadata
* empty content
* malformed JSON
* invalid UTF-8
* duplicate IDs
* duplicate URLs
* broken references
* inactive records accidentally being retrieved

---

# 28. JSON VALIDATION

Every final JSON file must:

* parse successfully
* use valid UTF-8
* have consistent schema
* contain unique IDs
* contain valid arrays
* contain valid strings/nulls
* contain no malformed JSON
* contain no accidental comments
* contain no trailing invalid syntax

Run an automated validation script.

---

# 29. DATA QUALITY REPORT

Create a report such as:

`data/DATA_QUALITY_REPORT.md`

Include:

* total source files
* total raw records
* total cleaned records
* duplicate records removed
* malformed records repaired
* irrelevant records removed
* records merged
* IELTS records
* country records
* service records
* contact records
* records with missing metadata
* records with missing source
* validation results
* unresolved issues

Do not hide unresolved issues.

---

# 30. SEARCH TEST SUITE

After cleaning the data, test at least:

### IELTS

1. can you give me the details of IELTS class?
2. tell me about IELTS coaching
3. what IELTS classes do you provide?
4. what facilities do IELTS students get?
5. who teaches IELTS?
6. are your IELTS faculty experienced?
7. are your IELTS faculty British Council trained?
8. do you provide IELTS study materials?
9. do you provide IELTS registration?
10. British Council IELTS
11. IDP IELTS
12. how can I register for IELTS?
13. tell me everything about IELTS
14. I want IELTS coaching
15. what support do you provide for IELTS?

### USA

16. study visa in USA
17. what is F1 visa?
18. what is M1 visa?
19. tell me about USA student visa

### Australia

20. student visa in Australia
21. tell me about Australian student visa

### Canada

22. visitor visa in Canada
23. work visa in Canada
24. study permit in Canada
25. LMIA in Canada

### Contact

26. email id
27. company email
28. phone number
29. mobile number
30. office address

### Company

31. who are you?
32. what is Precious Education?
33. what services do you provide?

### Unknown

34. what is your IELTS fee?
35. what are your IELTS class timings?
36. do you guarantee 8 bands?

For unknown information, do not invent answers.

---

# 31. CROSS-CONTEXT TESTING

These tests are mandatory.

Test:

User:
"I want IELTS coaching."

Then:
"tell me about USA student visa"

The second answer must NOT contain IELTS information unless relevant.

Test:

User:
"I need a Canada visitor visa."

Then:
"what about study visa?"

The system must correctly understand the new intent.

Test:

User:
"I need IELTS classes."

Then:
"how does it work?"

The system should maintain IELTS context.

---

# 32. REALISTIC CLIENT QUESTIONS

Do not test only exact keywords.

Test natural language:

* "Could you explain your IELTS coaching?"
* "I want to know more about the IELTS classes you offer."
* "What do you provide to students preparing for IELTS?"
* "Can you tell me about your IELTS faculty and facilities?"
* "Do you help students with IELTS registration?"
* "What kind of resources do you have for IELTS preparation?"
* "I am planning to take IELTS. How can Precious Education help?"

The retrieval system must work with these variations.

---

# 33. RESPONSE QUALITY TEST

For every major service, verify:

1. Correct topic
2. Correct country
3. Correct service
4. Correct facts
5. Complete relevant information
6. No hallucination
7. No unrelated information
8. Natural wording
9. Client-friendly answer
10. Source traceability

---

# 34. RESPONSE DIVERSITY

The same question may produce different wording.

For example:

Question:
"Tell me about IELTS coaching."

Response 1:
"Yes, Precious Education provides IELTS coaching..."

Response 2:
"Absolutely. Precious Education offers IELTS preparation support..."

Response 3:
"Yes. IELTS students can receive coaching from experienced..."

The facts must remain consistent.

Do not randomly alter factual information.

---

# 35. DO NOT HARD-CODE QUESTIONS

NEVER create:

```python
if query == "can you give me the details of IELTS class":
    return "..."
```

This is prohibited.

The system must generalize to unseen questions.

Data and retrieval must drive the answer.

---

# 36. WEBSITE SOURCE OF TRUTH

For current Precious Education website claims:

Official website knowledge has priority.

Do not replace website facts with model memory.

Do not allow the LLM to override retrieved official website facts.

For missing information:

do not guess.

---

# 37. FINAL DATA ARCHITECTURE

The final architecture should be:

RAW WEBSITE DATA
↓
RAW BACKUP
↓
DATA AUDIT
↓
NORMALIZATION
↓
DEDUPLICATION
↓
FACT EXTRACTION
↓
METADATA ENRICHMENT
↓
SOURCE PROVENANCE
↓
CANONICAL KNOWLEDGE DATA
↓
CHUNKING
↓
SEARCH INDEX
↓
RELEVANCE RANKING
↓
KNOWLEDGE ROUTER
↓
CONTEXT BUILDER
↓
CUSTOM LLM
↓
CLIENT RESPONSE

---

# 38. IMPORTANT: DO NOT MODIFY UNRELATED AI COMPONENTS UNNECESSARILY

This task is primarily about:

* data
* data quality
* schema
* website knowledge
* retrieval compatibility
* knowledge routing
* context quality

Do not rebuild:

* tokenizer
* Transformer architecture
* training pipeline
* MongoDB architecture
* PHP frontend

unless inspection proves that a change is absolutely required for the cleaned knowledge data to function.

---

# 39. BACKWARD COMPATIBILITY

The cleaned data must remain compatible with the existing application.

Before changing schemas:

inspect all code that consumes the current JSON files.

Update dependent code where necessary.

Do not clean the JSON files in a way that silently breaks:

* ingestion
* retrieval
* API
* Knowledge Router
* Context Builder
* LLM
* tests
* PHP frontend

---

# 40. FINAL ACCEPTANCE CRITERIA

Do not claim the task is complete merely because the JSON files parse.

The task is complete only when:

[ ] All data files audited

[ ] Raw data backed up

[ ] All useful information preserved

[ ] Duplicate information cleaned

[ ] Malformed data repaired

[ ] Schema normalized

[ ] Metadata normalized

[ ] Source provenance preserved

[ ] IELTS completely reconstructed

[ ] All service data cleaned

[ ] All country data cleaned

[ ] Contact data isolated

[ ] Company identity data isolated

[ ] Country/service isolation works

[ ] Retrieval returns multiple relevant chunks

[ ] Details queries return comprehensive information

[ ] Unknown information is not hallucinated

[ ] Follow-up context works

[ ] Existing custom LLM remains in use

[ ] No external AI APIs introduced

[ ] JSON validation passes

[ ] Retrieval tests pass

[ ] End-to-end `/api/chat` tests pass

[ ] PHP → FastAPI integration passes

[ ] Regression tests pass

---

# 41. FINAL REPORT

At the end provide:

## DATA RECONSTRUCTION SUMMARY

* files inspected
* files modified
* files created
* files backed up
* records before
* records after
* duplicates removed
* records merged
* malformed records fixed
* unresolved records

## IELTS SUMMARY

Show the actual IELTS knowledge categories/facts discovered and retained.

## SERVICES SUMMARY

List all cleaned service categories.

## COUNTRY SUMMARY

List all cleaned countries and their available information.

## CONTACT SUMMARY

List the contact-related knowledge records that are available.

## RETRIEVAL SUMMARY

Show sample queries and retrieved records.

## END-TO-END TEST SUMMARY

Show PASS/FAIL for every required test.

## REMAINING ISSUES

Clearly list anything that could not be fixed because the source data itself does not contain the required information.

DO NOT CLAIM "100% PERFECT" unless the actual validation supports that conclusion.

The objective is not to make the dataset look perfect.

The objective is to make it:

ACCURATE
COMPLETE
CONSISTENT
SOURCE-GROUNDED
SEARCHABLE
TRACEABLE
NON-HALLUCINATORY
CLIENT-READY
AND COMPATIBLE WITH THE EXISTING PRECIOUS AI SYSTEM.
















Next Prompt :- 

# PRECIOUS AI — REAL-TIME CHATGPT-LIKE EXPERIENCE

# MASTER QUALITY, ACCURACY, DATA, RETRIEVAL & RESPONSE FIX

You are working on the existing Precious AI system for Precious Education.

The objective of this task is to make the chatbot feel like a polished, real-time conversational AI assistant while remaining fully grounded in the actual Precious Education data.

This is an EXISTING CUSTOM AI SYSTEM.

DO NOT replace it with an external AI API.

DO NOT use:

* OpenAI API
* ChatGPT API
* Gemini
* Claude
* Anthropic
* Groq
* OpenRouter
* external inference APIs
* external embeddings
* external vector databases
* pretrained LLMs
* pretrained tokenizers

The system must continue using the existing:

* Custom tokenizer
* Custom Transformer/LLM
* Custom inference
* Conversation Engine
* Memory/context system
* Website Knowledge Engine
* Project/Excel Knowledge Engine
* MongoDB
* FastAPI
* PHP frontend

You may modify existing JSON data files and supporting code where necessary.

DO NOT rebuild components unnecessarily.

The goal is:

# NATURAL CONVERSATION + CORRECT KNOWLEDGE + COMPLETE ANSWERS + CONTEXT AWARENESS + CONTROLLED DIVERSITY + NO HALLUCINATION

---

# 1. FINAL USER EXPERIENCE

Precious AI should feel like a professional conversational assistant.

The user should be able to talk naturally.

Examples:

User:
"Hi"

Assistant:
"Hello! How can I help you today?"

User:
"My name is Ritesh."

Assistant:
"Nice to meet you, Ritesh! How can I help you with your study abroad or immigration plans?"

User:
"I need a student visa for USA."

Assistant:
Provide USA-specific student visa information from the available knowledge.

User:
"What is F1?"

Assistant:
Explain F1 in the context of USA study.

User:
"What about M1?"

Assistant:
Understand that M1 refers to the same USA student-visa topic.

User:
"Can you tell me about IELTS classes?"

Assistant:
Provide complete IELTS information available in the website knowledge.

User:
"How does it work?"

Assistant:
Understand that "it" refers to IELTS classes, not start a new unrelated answer.

User:
"Okay thanks."

Assistant:
Respond naturally and politely.

The system must not feel like a collection of hard-coded FAQ responses.

---

# 2. ABSOLUTE ACCURACY RULE

The most important priority is:

FACTUAL ACCURACY > COMPLETENESS > NATURAL LANGUAGE > RESPONSE DIVERSITY

Never sacrifice factual correctness for creativity.

The LLM may change:

* wording
* sentence structure
* opening
* bullet ordering
* explanation style

But it must NOT change:

* facts
* country
* service
* visa category
* contact information
* company information
* source claims

---

# 3. COMPLETE DATA AUDIT

Inspect the entire:

`data/`

directory.

Inspect ALL JSON files recursively.

Do not assume one file contains all information.

Audit:

* services
* countries
* visa categories
* IELTS
* company information
* contact information
* about information
* admissions
* immigration
* work permits
* visit visas
* inadmissibility
* other services
* website pages
* website chunks
* metadata
* source URLs

Find:

* duplicates
* missing information
* malformed JSON
* conflicting records
* incorrect categories
* incorrect country labels
* fragmented information
* incomplete chunks
* repeated content
* irrelevant content
* poor metadata

---

# 4. DATA FILES MAY BE MODIFIED

You are explicitly allowed to:

* edit JSON
* merge JSON records
* split JSON records
* normalize schemas
* remove duplicates
* fix malformed content
* reorganize records
* create new normalized JSON files
* create derived search indexes

BUT:

Never destroy the original data without backup.

Create a raw backup first.

---

# 5. SOURCE-GROUNDED DATA

Every factual record should retain provenance where possible:

* source URL
* source file
* source page
* source title
* source section
* content hash
* version
* active status

The system must be able to answer:

"Where did this information come from?"

internally.

---

# 6. NO FABRICATION

If information exists:

USE IT.

If information does not exist:

DO NOT INVENT IT.

Examples:

If IELTS fee is unavailable:

Do not invent a fee.

If IELTS class timing is unavailable:

Do not invent timings.

If visa processing time is unavailable:

Do not invent processing time.

If company owner information is unavailable:

Do not guess.

If a visa requirement is unavailable:

Do not manufacture one.

Use a natural response such as:

"I don't currently have that information in the available Precious Education data."

---

# 7. IELTS MUST BE COMPLETE

Find ALL IELTS information across all source JSON files.

Do not rely on one IELTS record.

Combine all unique IELTS information.

Examples of potentially relevant information:

* IELTS coaching
* IELTS classes
* experienced faculty
* British Council-trained faculty
* personalized coaching
* preparatory materials
* CDs
* library
* brochures
* video tapes
* British Council registration
* IDP registration
* authorized registration center
* any other IELTS information actually present in the source

Preserve all unique facts.

Do not compress them into:

"We provide IELTS training."

---

# 8. CANONICAL KNOWLEDGE STRUCTURE

Create a consistent internal representation.

Use fields such as:

```json
{
  "knowledge_id": "",
  "knowledge_type": "",
  "category": "",
  "subcategory": "",
  "country": null,
  "service": "",
  "topic": "",
  "title": "",
  "content": "",
  "facts": [],
  "keywords": [],
  "synonyms": [],
  "source_url": "",
  "source_file": "",
  "source_section": "",
  "source_title": "",
  "content_hash": "",
  "version": "",
  "active": true
}
```

Do not fill missing fields with fake information.

---

# 9. KNOWLEDGE CATEGORIES

Support at least:

* company
* about
* service
* country
* visa
* IELTS
* admissions
* immigration
* work
* visit
* inadmissibility
* contact
* project
* other

Use actual source data to populate them.

---

# 10. QUERY UNDERSTANDING

The chatbot must understand natural language.

Do not depend on exact keywords.

Examples:

"Can you tell me about IELTS?"

"Could you give me details about your IELTS classes?"

"I want information about IELTS coaching."

"What support do you offer for IELTS preparation?"

"What facilities are available for IELTS students?"

All should identify IELTS as the main topic.

---

# 11. ENTITY DETECTION

Detect relevant entities such as:

Countries:

* USA
* Canada
* Australia
* UK
* New Zealand
* Singapore
* Malaysia
* Ireland

Services:

* IELTS
* Study Visa
* Student Visa
* Work Visa
* Visitor Visa
* Immigration
* Admissions
* LMIA
* PNP
* etc.

Visa categories:

* F1
* M1
* F2
* M2
* Study Permit
* PGWP
* LMIA
* etc.

Do not confuse related but different concepts.

---

# 12. COUNTRY ISOLATION

This is mandatory.

If user says:

"Study visa in USA"

USA information must dominate.

Do not answer with:

Canada SDS
Canada PNP
Canada Study Permit

unless relevant.

If user says:

"Visitor visa in Canada"

retrieve Canada visitor-visa information.

Do not return Canada student-visa information simply because "Canada" and "visa" match.

If user says:

"Student visa in Australia"

retrieve Australia information.

---

# 13. SERVICE ISOLATION

If user asks:

"IELTS"

prioritize IELTS.

Do not return:

* Canada immigration
* USA F1
* work permits
* visitor visa

just because they are common website content.

If user asks:

"LMIA"

prioritize LMIA/work-visa information.

If user asks:

"email"

prioritize contact information.

---

# 14. CONTACT ROUTING

Queries:

* email
* email ID
* email address
* contact email
* phone
* mobile number
* contact number
* office address
* location
* contact details

must route to dedicated contact knowledge.

Do not answer with generic service information.

---

# 15. COMPANY IDENTITY ROUTING

Queries:

* who are you?
* what is Precious Education?
* tell me about the company
* what does your company do?

must retrieve company/about information.

For owner/founder questions:

Only answer if explicit source information exists.

Never guess.

---

# 16. RETRIEVAL MUST BE MULTI-CHUNK

For broad questions:

"Tell me about IELTS."

"Give me details about your services."

"What can you help me with?"

Retrieve multiple relevant chunks.

Suggested:

TOP_K = 5–8

Then apply relevance threshold.

Do not simply take the first search result.

---

# 17. RELEVANCE SCORING

Use deterministic local scoring.

Suggested:

Exact phrase = +15
Exact topic = +12
Exact service = +12
Exact country = +12
Keyword = +8
Title = +8
Heading = +7
Section = +6
Synonym = +3
Multiple keyword overlap = +5

Penalties:

Wrong country = -15
Wrong service = -15
Wrong visa category = -15
Unrelated category = -10
Very weak overlap = -8

The final ranking must prioritize semantic relevance using deterministic signals.

---

# 18. QUERY EXPANSION

Support simple deterministic normalization.

Examples:

"USA"
"US"
"United States"
"United States of America"

can map to a normalized country identifier where appropriate.

Similarly:

"IELTS class"
"IELTS classes"
"IELTS coaching"
"IELTS training"
"IELTS preparation"

can map to the IELTS topic.

Do not change factual meaning.

---

# 19. BROAD QUESTION DETECTION

Recognize:

* details
* information
* tell me about
* explain
* what do you offer
* what services
* everything about
* overview
* facilities
* support

as broad informational requests.

For these requests:

Retrieve multiple relevant facts.

Generate a sufficiently complete answer.

---

# 20. PROCESS QUESTIONS

Recognize:

* how
* process
* procedure
* steps
* what should I do
* how can I apply
* how can I get
* what happens next

as process-oriented questions.

Use process information from the knowledge base if available.

Do not invent a process.

If process information is incomplete:

clearly distinguish available information from information not present in the knowledge base.

---

# 21. CONVERSATION CONTEXT

The system must behave conversationally.

Example:

User:
"I want IELTS coaching."

Assistant:
IELTS answer.

User:
"How does it work?"

The system must understand:

"it" = IELTS coaching.

User:
"Do you provide registration?"

The system should continue the IELTS context.

User:
"What about British Council?"

Still IELTS context.

Do not lose context after every message.

---

# 22. CONTEXT WINDOW MANAGEMENT

Do not send the entire conversation history indefinitely.

Maintain useful recent context.

Prioritize:

1. Current query
2. Current topic
3. Relevant previous turns
4. User preferences/name where appropriate
5. Retrieved knowledge
6. Older conversation only when relevant

Prevent irrelevant context from contaminating the answer.

---

# 23. CONTEXT ISOLATION

Sessions must be isolated.

User A's information must never appear in User B's conversation.

Project knowledge must not leak between unrelated queries.

Website knowledge should only be included when relevant.

---

# 24. LLM CONTEXT FORMAT

Construct a clean context for the custom LLM:

SYSTEM INSTRUCTIONS

CONVERSATION CONTEXT

CURRENT USER QUESTION

DETECTED INTENT

DETECTED ENTITIES

RELEVANT WEBSITE KNOWLEDGE

RELEVANT PROJECT KNOWLEDGE

RESPONSE REQUIREMENTS

The model must clearly understand which information is authoritative.

---

# 25. KNOWLEDGE PRIORITY

Use this priority:

1. Current retrieved official website/project information
2. Relevant conversation context
3. Custom domain-trained knowledge
4. General model knowledge

For current business facts, retrieved official knowledge has priority.

---

# 26. ANSWER GENERATION

The answer must:

* directly answer the user
* use relevant retrieved information
* include major available facts
* be natural
* be concise when the question is simple
* be detailed when the question asks for details
* use bullets when appropriate
* avoid unnecessary repetition
* avoid robotic phrases
* avoid generic filler
* avoid irrelevant URLs

Do not answer every question with the same structure.

---

# 27. EXAMPLE — IELTS

User:

"Can you give me the details of IELTS class?"

If the knowledge base contains the relevant facts, a high-quality answer should be similar in completeness to:

"Yes, we provide expert IELTS coaching services at Precious Education! Here are the details of our IELTS class:

* Experienced Faculty: Learn from experienced and British Council-trained faculty members who ensure the right input is imparted to students.
* Personalized Coaching: Personalized coaching is supported by preparatory materials, CDs, a well-equipped library containing brochures and video tapes, and dedicated study resources.
* Authorized Registration Center: Precious Education is an authorized center to accept registrations for both British Council and IDP IELTS exams.

If you would like to know more about the available IELTS coaching or registration support, feel free to ask."

IMPORTANT:

Do NOT hard-code this answer.

Generate it from the actual retrieved knowledge.

If some of these facts are absent from the current source data, do not invent them.

---

# 28. NATURAL LANGUAGE QUALITY

Avoid repetitive openings like:

"According to our details..."

"To answer your question..."

"Okay, here is the information..."

"Our team says..."

"Yes! We provide..."

These should NOT appear in every response.

Prefer natural conversational language.

---

# 29. RESPONSE LENGTH

Implement dynamic response length.

Simple question:

"what is F1?"

→ short explanation.

Broad question:

"tell me everything about IELTS coaching"

→ detailed answer.

Contact question:

"email?"

→ direct answer.

Process question:

"how can I apply?"

→ structured steps if supported.

Do not use one fixed response length for everything.

---

# 30. RESPONSE DIVERSITY

Enable controlled generation diversity if supported by the existing custom LLM.

Possible generation controls:

temperature
top-k
top-p
repetition penalty

Start with conservative settings.

The goal is:

different wording

NOT

different facts.

For the same question, multiple valid answers may vary in:

* opening
* sentence structure
* bullet phrasing
* ordering
* explanation style

But facts must remain stable.

---

# 31. REPETITION CONTROL

Detect:

* repeated sentences
* repeated phrases
* repeated answers
* loops
* token repetition

Do not allow the model to generate:

"The IELTS class..."
"The IELTS class..."
"The IELTS class..."

or repeat the same paragraph.

Use:

* repetition penalty
* no-repeat n-gram logic where compatible
* EOS handling
* maximum generation length
* response deduplication

---

# 32. EOS / TERMINATION

Ensure the custom LLM stops naturally.

It must not:

* endlessly generate
* repeat text
* produce broken fragments
* continue after the answer is complete

Verify tokenizer EOS behavior.

Verify inference stopping conditions.

---

# 33. REAL-TIME EXPERIENCE

The chatbot should feel responsive.

Optimize:

* request handling
* retrieval latency
* context construction
* MongoDB queries
* model inference
* unnecessary repeated database work

If the existing architecture supports streaming safely, implement SSE/token streaming.

If streaming is not currently stable, do not sacrifice correctness for streaming.

The priority is:

fast + correct + stable.

---

# 34. RESPONSE POST-PROCESSING

Inspect post-processing.

Do not blindly modify generated answers.

Post-processing may safely:

* remove obvious duplicate sentences
* normalize whitespace
* remove accidental repeated punctuation
* clean malformed output

It must NOT:

* rewrite facts
* inject hard-coded answers
* change country names
* add unsupported information

---

# 35. SOURCE LINKS

If source URLs are available:

Answer the question first.

Then optionally provide a relevant source link.

Do not use:

"For more details, visit the website."

as a replacement for the actual answer.

---

# 36. UNKNOWN / NO MATCH

If retrieval finds no relevant knowledge:

Do not use a random generic answer.

Do not retrieve unrelated content.

Return a useful natural response explaining that the specific information is not currently available.

Example:

"I don't currently have that specific information in the available Precious Education knowledge. If you'd like, I can help with the information I do have about IELTS coaching and registration."

Only say what is supported.

---

# 37. HALLUCINATION PROTECTION

For every generated response, enforce:

CLAIM → SOURCE SUPPORT

Every factual business claim should be traceable to:

* retrieved website knowledge
* retrieved project knowledge
* explicitly supported conversation context

If a claim has no support and is not basic conversational language:

do not generate it.

---

# 38. PROJECT KNOWLEDGE SEPARATION

Website knowledge and project/Excel knowledge are separate.

Website:

current business/service/company information.

Project:

live project-specific records.

Do not mix them.

For example:

"What is your IELTS coaching?"

→ Website knowledge.

"What is the status of Project ABC?"

→ Project knowledge.

"What is Project ABC and what services does Precious Education provide?"

→ Both, if relevant.

---

# 39. DATABASE DESIGN

Inspect MongoDB collections.

Ensure knowledge records can be efficiently queried by:

* topic
* service
* country
* category
* keywords
* active status
* source
* version

Add indexes where genuinely useful.

Do not introduce an external vector database.

---

# 40. DEVELOPMENT DEBUGGING

Add a development-only debug capability that shows:

Query

→ normalized query

→ intent

→ entities

→ country

→ service

→ retrieval results

→ scores

→ selected chunks

→ final context

→ generated answer

This is mandatory for diagnosing wrong answers.

Never expose internal debug information to normal clients.

---

# 41. CRITICAL REGRESSION TESTS

Test all of these:

### General

"Hi"

"Hello"

"Bye"

"Thanks"

"My name is Ritesh"

"Who are you?"

### IELTS

"Can you give me the details of IELTS class?"

"Tell me about IELTS coaching."

"What facilities do you provide for IELTS?"

"Who teaches IELTS?"

"Do you provide IELTS registration?"

"British Council IELTS"

"IDP IELTS"

"What study resources do you provide?"

### USA

"Study visa in USA"

"What is F1?"

"What is M1?"

"How can I get a student visa in USA?"

### Australia

"Student visa in Australia"

### Canada

"Visitor visa in Canada"

"Work visa in Canada"

"LMIA"

### Contact

"What is your email?"

"What is your phone number?"

"What is your office address?"

### Unknown

"What is your IELTS fee?"

"What are your IELTS class timings?"

"Do you guarantee an 8 band score?"

Do not fabricate answers to unavailable questions.

---

# 42. MULTI-TURN TESTS

Test:

User:
"I want IELTS coaching."

Assistant:
IELTS response.

User:
"How does it work?"

Assistant:
IELTS-related response.

User:
"Do you provide registration?"

Assistant:
IELTS registration response.

User:
"What about British Council?"

Assistant:
IELTS/British Council response.

Then:

User:
"Okay thanks."

Assistant:
Natural closing.

---

# 43. CROSS-CONTEXT TEST

Test:

User:
"I want IELTS coaching."

Then:

"Tell me about USA student visa."

The response must switch to USA.

Then:

"What is F1?"

The response must understand USA context.

Then:

"How about Canada?"

The response must switch to Canada.

No stale IELTS context should contaminate the answer.

---

# 44. SAME-QUESTION DIVERSITY TEST

Run:

"Tell me about IELTS coaching."

at least 10 times.

Measure:

* unique responses
* exact duplicates
* repeated phrases
* factual consistency
* unsupported claims
* response length

Target:

Natural wording variation.

ZERO factual drift.

ZERO country drift.

ZERO service drift.

ZERO hallucinated claims.

---

# 45. DATA-DRIVEN TEST

Create tests that use questions NOT explicitly present in the dataset.

Examples:

"Could you explain the IELTS support you offer?"

"I am planning to take IELTS. What help can I get?"

"What resources are available for IELTS preparation?"

"Can your center help with IELTS registration?"

"What kind of faculty teaches IELTS?"

The system must answer through retrieval + context + custom LLM.

Do not hard-code these questions.

---

# 46. NO FAQ HARD-CODING

This is strictly prohibited:

```python
if query == "...":
    return "..."
```

Do not build hundreds of question-specific rules.

Broad intent/entity routing is acceptable.

Individual question-answer hard-coding is NOT.

---

# 47. MODEL QUALITY AUDIT

Inspect the actual custom LLM.

Verify:

* correct tokenizer is used
* correct checkpoint is loaded
* trained weights are loaded
* model is not accidentally initialized randomly
* inference uses trained model
* EOS works
* generation works
* context is actually passed into the model
* maximum context is sufficient
* output length is sufficient
* generation parameters are reasonable

If the model is failing despite receiving correct context, identify whether the limitation is:

* tokenizer
* context length
* model capacity
* instruction following
* training quality
* inference
* decoding

Do not hide model limitations with hard-coded responses.

---

# 48. IMPORTANT: DO NOT TRAIN BLINDLY

Do not retrain the entire LLM simply because one response is bad.

First determine:

Is the data correct?

Is retrieval correct?

Is context correct?

Is the prompt correct?

Is the model receiving the context?

Is generation working?

Only then determine whether additional model training is genuinely required.

---

# 49. PERFORMANCE

Measure:

* retrieval latency
* MongoDB query latency
* context-building latency
* model inference latency
* total `/api/chat` latency

Optimize obvious bottlenecks.

Do not trade accuracy for meaningless micro-optimizations.

---

# 50. PHP → FASTAPI → LLM END-TO-END

Test the complete path:

PHP frontend
↓
FastAPI
↓
Conversation Engine
↓
Intent Detection
↓
Knowledge Router
↓
Website Retrieval
↓
Project Retrieval if needed
↓
Context Builder
↓
Custom LLM
↓
Response Post-Processing
↓
MongoDB conversation storage
↓
FastAPI response
↓
PHP display

Do not test only isolated functions.

---

# 51. ERROR HANDLING

If:

* MongoDB is unavailable
* knowledge retrieval fails
* model fails
* malformed data is found
* context is empty
* inference times out

the API must return a controlled error/fallback.

Never expose:

* stack traces
* internal paths
* database credentials
* internal prompts
* debug information

to the client.

---

# 52. FINAL QUALITY GATE

Do not claim success just because the application starts.

The following must actually pass:

DATA QUALITY
PASS/FAIL

JSON VALIDATION
PASS/FAIL

IELTS COMPLETENESS
PASS/FAIL

SERVICE COMPLETENESS
PASS/FAIL

COUNTRY ISOLATION
PASS/FAIL

CONTACT ROUTING
PASS/FAIL

COMPANY ROUTING
PASS/FAIL

MULTI-CHUNK RETRIEVAL
PASS/FAIL

CONTEXT QUALITY
PASS/FAIL

CUSTOM LLM GROUNDING
PASS/FAIL

HALLUCINATION CONTROL
PASS/FAIL

FOLLOW-UP CONTEXT
PASS/FAIL

RESPONSE DIVERSITY
PASS/FAIL

REPETITION CONTROL
PASS/FAIL

EOS TERMINATION
PASS/FAIL

API
PASS/FAIL

PHP INTEGRATION
PASS/FAIL

END-TO-END CHAT
PASS/FAIL

---

# 53. FINAL REPORT

Provide:

## DATA

* files inspected
* files changed
* files created
* records before
* records after
* duplicates removed
* malformed records fixed
* missing information identified

## RETRIEVAL

Show sample queries and actual retrieved chunks/scores.

## LLM

Show:

* model/checkpoint used
* tokenizer used
* generation configuration
* context length
* EOS behavior

## CHAT QUALITY

Show actual input/output examples.

## DIVERSITY

Show repeated-query statistics.

## HALLUCINATION

Show unknown-information tests.

## PERFORMANCE

Show latency measurements.

## REMAINING ISSUES

Clearly state anything that could not be solved because the underlying source data or model capability is insufficient.

---

# FINAL OBJECTIVE

Precious AI must NOT feel like:

"keyword search + template response."

It must feel like:

A real conversational assistant that understands what the client is asking, remembers the current conversation, retrieves the correct Precious Education information, combines relevant facts, answers naturally, avoids unsupported claims, and responds differently when appropriate while keeping the facts exactly consistent.

The desired pipeline is:

USER
↓
CONVERSATION UNDERSTANDING
↓
INTENT + ENTITY DETECTION
↓
CONTEXT RESOLUTION
↓
KNOWLEDGE ROUTER
↓
WEBSITE / PROJECT RETRIEVAL
↓
RELEVANCE RANKING
↓
MULTI-CHUNK CONTEXT
↓
CUSTOM LLM
↓
CONTROLLED GENERATION
↓
REPETITION / SAFETY CHECK
↓
FINAL CLIENT-READY RESPONSE

The final response must be:

NATURAL
ACCURATE
GROUNDED
COMPLETE
CONTEXT-AWARE
NON-REPETITIVE
CLIENT-FRIENDLY
FAST
STABLE
AND DATA-DRIVEN.

Do not claim perfection without executing the tests.
