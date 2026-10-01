"""Generate diverse, richly-detailed persona cards for simulation agents via LLM.

Each persona includes backstory, speaking mannerisms, hobbies, current concerns,
values, and relationship hints — giving the LLM enough character depth to produce
distinctly-voiced, anaphora-ready dialogue.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

from MASim.core.schema import PersonaCard
from MASim.generation._hint_profiles import HINT_PROFILES
from MASim.ground_truth.json_parser import clean_llm_json_text, parse_json_object
from MASim.prompts import PERSONA_SYSTEM as PERSONA_SYSTEM_PROMPT
from MASim.prompts import PERSONA_USER as PERSONA_USER_TEMPLATE
from MASim.utils.logging import get_logger

log = get_logger(__name__)


def _repair_json(text: str) -> dict:
    """Try json.loads; on failure, attempt common repairs and retry."""
    try:
        return parse_json_object(text)
    except (ValueError, TypeError):
        pass
    text = clean_llm_json_text(text)
    # Strip trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single quotes with double quotes (crude but effective for LLM output)
    # Only if there are no double quotes at all or single-quote keys are detected
    if "'" in fixed and fixed.count('"') < fixed.count("'"):
        fixed = fixed.replace("'", '"')
    # Remove control characters
    fixed = re.sub(r"[\x00-\x1f]+", " ", fixed)
    return json.loads(fixed)


DIVERSITY_HINTS = [
    # ── R — Realistic (0-33) ─────────────────────────────────────────────────
    "[R] electrician who reads architecture books in the evenings",        # 0
    "[R] auto mechanic transitioning to electric vehicles",               # 1
    "[R] structural welder on commercial building projects",              # 2
    "[R] HVAC technician building a small business",                      # 3
    "[R] landscape gardener designing community green spaces",            # 4
    "[R] furniture maker specializing in reclaimed wood",                 # 5
    "[R] stone mason restoring heritage buildings",                       # 6
    "[R] long-haul truck driver considering retirement",                  # 7
    "[R] beekeeper supplying local restaurants and shops",                # 8
    "[R] city arborist managing the urban tree canopy",                   # 9
    "[R] locksmith and home security consultant",                         # 10
    "[R] sous chef at a fine-dining restaurant",                          # 11
    "[R] semi-pro athlete coaching youth sports",                         # 12
    "[R] wildfire prevention specialist in drought country",              # 13
    "[R] solar panel installer turned site supervisor",                   # 14
    "[R] marine engineer on oceanographic research vessels",              # 15
    "[R] sailboat restorer and youth sailing coach",                      # 16
    "[R] blacksmith crafting custom knives and tools",                    # 17
    "[R] glass blower with a small studio gallery",                       # 18
    "[R] agricultural drone operator surveying farmland",                 # 19
    "[R] ex-military logistics officer in civilian supply chains",        # 20
    "[R] park maintenance supervisor in a large metro area",              # 21
    "[R] general contractor building affordable housing",                 # 22
    "[R] harbour pilot navigating container ships",                       # 23
    "[R] bicycle courier in a busy capital city",                         # 24
    "[R] watch repairer preserving a disappearing trade",                 # 25
    "[R] brewery operations manager and avid homebrewer",                 # 26
    "[R] veterinary technician at a rural large-animal practice",         # 27
    "[R] deep-sea commercial diver",                                      # 28
    "[R] long-distance train conductor",                                  # 29
    "[R] helicopter rescue pilot in mountain terrain",                    # 30
    "[R] prosthetic limb technician fitting custom devices",              # 31
    "[R] ferry captain on a daily commuter route",                        # 32
    "[R] industrial electrician in a manufacturing plant",                # 33
    # ── I — Investigative (34-66) ────────────────────────────────────────────
    "[I] young tech professional recently relocated to a new city",       # 34
    "[I] retired academic writing a history newsletter",                  # 35
    "[I] PhD student six years into a niche dissertation",                # 36
    "[I] marine biologist tracking coral reef decline",                   # 37
    "[I] data analyst who moonlights as a stand-up comedian",             # 38
    "[I] first-generation university student from a farming town",        # 39
    "[I] forensic scientist in a state crime lab",                        # 40
    "[I] geologist consulting for mining, ethically conflicted",          # 41
    "[I] epidemiologist recovering from pandemic-era burnout",            # 42
    "[I] astrophysicist doing public science outreach",                   # 43
    "[I] conservation biologist monitoring endangered species",            # 44
    "[I] agricultural researcher developing drought-resistant crops",     # 45
    "[I] field archaeologist directing a small excavation",               # 46
    "[I] cybersecurity analyst at a major bank",                          # 47
    "[I] machine learning researcher at a health non-profit",             # 48
    "[I] environmental chemist testing municipal water supplies",         # 49
    "[I] biomedical researcher in a stalled clinical trial",              # 50
    "[I] urban ecologist studying city wildlife adaptation",              # 51
    "[I] sports physiologist at a university research lab",               # 52
    "[I] nuclear safety inspector approaching retirement",                # 53
    "[I] forensic accountant investigating corporate fraud",              # 54
    "[I] pharmacist at a community drugstore",                            # 55
    "[I] medical lab technician on the night shift",                      # 56
    "[I] radiographer at a teaching hospital",                            # 57
    "[I] veterinarian running a rural mixed practice",                    # 58
    "[I] climate modeller at a government research institute",            # 59
    "[I] patent examiner with an engineering background",                 # 60
    "[I] food scientist developing plant-based proteins",                 # 61
    "[I] neuropsychologist researching memory in older adults",           # 62
    "[I] soil scientist advising on regenerative agriculture",            # 63
    "[I] actuary at an insurance firm, bored but well-paid",             # 64
    "[I] volcanologist stationed at a remote observatory",                # 65
    "[I] robotics engineer building assistive devices",                   # 66
    # ── A — Artistic (67-99) ─────────────────────────────────────────────────
    "[A] working musician struggling with financial instability",         # 67
    "[A] investigative journalist chasing a local corruption story",      # 68
    "[A] freelance translator working across three languages",            # 69
    "[A] architect pivoting to sustainable community design",             # 70
    "[A] librarian who runs a neighbourhood zine collective",             # 71
    "[A] documentary filmmaker on a self-funded personal project",        # 72
    "[A] theatre director at a small regional repertory company",         # 73
    "[A] graphic novelist with a cult online following",                  # 74
    "[A] ceramicist selling at weekend craft markets",                    # 75
    "[A] fashion designer launching a sustainable clothing label",        # 76
    "[A] live sound engineer for touring bands",                          # 77
    "[A] street photographer preparing a debut gallery show",             # 78
    "[A] podcast producer covering true-crime stories",                   # 79
    "[A] tattoo artist with a classical fine-arts education",             # 80
    "[A] stained-glass restorer for historic churches",                   # 81
    "[A] costume designer for independent film productions",              # 82
    "[A] typeface designer working remotely from a small town",           # 83
    "[A] mystery novelist with a day job in insurance",                   # 84
    "[A] poet on a one-year university residency",                        # 85
    "[A] children's book illustrator and full-time parent",               # 86
    "[A] community dance teacher and choreographer",                      # 87
    "[A] calligrapher and antique book restorer",                         # 88
    "[A] indie game designer working solo",                               # 89
    "[A] mural painter doing public-art commissions",                     # 90
    "[A] film editor at a small documentary studio",                      # 91
    "[A] jazz pianist playing hotel lobbies and cocktail bars",           # 92
    "[A] science writer at a national newspaper",                         # 93
    "[A] advertising copywriter who recently went freelance",             # 94
    "[A] independent press book editor",                                  # 95
    "[A] photojournalist documenting migration stories",                  # 96
    "[A] heritage tour guide who writes local history",                   # 97
    "[A] orchestra conductor in a mid-size city",                         # 98
    "[A] music therapist at a children's hospital",                       # 99
    # ── S — Social (100-133) ─────────────────────────────────────────────────
    "[S] emergency department nurse on rotating shifts",                  # 100
    "[S] stay-at-home parent re-entering the workforce",                 # 101
    "[S] public defender handling an impossible caseload",                # 102
    "[S] social worker in a high-poverty urban neighbourhood",           # 103
    "[S] high school English teacher near burnout",                      # 104
    "[S] hospice nurse who volunteers with refugee families",            # 105
    "[S] community organiser in a gentrifying neighbourhood",            # 106
    "[S] recently-divorced parent navigating co-parenting",              # 107
    "[S] midwife at a community birthing centre",                        # 108
    "[S] occupational therapist for stroke recovery patients",           # 109
    "[S] school counselor in an underfunded district",                   # 110
    "[S] ESL instructor working with adult immigrants",                  # 111
    "[S] special education coordinator fighting for resources",          # 112
    "[S] refugee resettlement caseworker",                                # 113
    "[S] substance abuse counselor in recovery themselves",              # 114
    "[S] youth programme coordinator for after-school activities",       # 115
    "[S] homeless shelter manager during a housing crisis",              # 116
    "[S] disability rights advocate who uses a wheelchair",              # 117
    "[S] speech-language pathologist in a public school",                # 118
    "[S] physical therapist in a sports rehabilitation clinic",          # 119
    "[S] dental hygienist studying for a dental degree at night",        # 120
    "[S] preschool teacher passionate about play-based learning",        # 121
    "[S] museum educator designing hands-on community exhibits",         # 122
    "[S] vocational trainer for displaced factory workers",              # 123
    "[S] literacy volunteer teaching adults in a prison programme",      # 124
    "[S] parish priest wrestling with doubt",                             # 125
    "[S] university interfaith chaplain",                                 # 126
    "[S] yoga instructor managing chronic back pain",                    # 127
    "[S] boxing gym owner in a struggling neighbourhood",                # 128
    "[S] rock climbing guide and outdoor educator",                      # 129
    "[S] dog trainer specializing in rescue animals",                    # 130
    "[S] therapeutic horse trainer at a riding centre",                  # 131
    "[S] sign language interpreter in courtrooms",                       # 132
    "[S] home health aide supporting elderly patients",                  # 133
    # ── E — Enterprising (134-166) ───────────────────────────────────────────
    "[E] small restaurant owner rebuilding after the pandemic",          # 134
    "[E] financial planner helping small business owners",               # 135
    "[E] accountant who recently started their own practice",            # 136
    "[E] real estate agent in a cooling market",                         # 137
    "[E] supply chain manager navigating global disruptions",            # 138
    "[E] independent bookshop owner near a university",                  # 139
    "[E] management consultant rethinking corporate culture",            # 140
    "[E] cross-border trade broker between two continents",              # 141
    "[E] organic farmer selling at weekend markets",                     # 142
    "[E] food truck owner building a local brand",                       # 143
    "[E] third-generation winemaker at a family vineyard",               # 144
    "[E] specialty coffee roaster and former barista",                   # 145
    "[E] concert and festival promoter in a mid-size city",              # 146
    "[E] vinyl record shop owner and weekend DJ",                        # 147
    "[E] antique dealer with an encyclopedic memory",                    # 148
    "[E] florist specializing in event design",                          # 149
    "[E] insurance claims adjuster weary of disaster work",              # 150
    "[E] credit union manager in a rural community",                     # 151
    "[E] used car dealer trying to build a trustworthy brand",           # 152
    "[E] property manager juggling dozens of tenants",                   # 153
    "[E] franchise restaurant manager saving to open their own place",   # 154
    "[E] wedding planner in a competitive city market",                  # 155
    "[E] recruitment consultant in the healthcare sector",               # 156
    "[E] small-town radio station owner",                                # 157
    "[E] independent travel agent specializing in adventure trips",      # 158
    "[E] artisan cheese maker at a small dairy",                         # 159
    "[E] craft brewery taproom manager",                                  # 160
    "[E] personal trainer building an online coaching brand",            # 161
    "[E] import shop owner selling goods from their home country",       # 162
    "[E] bed-and-breakfast operator in a tourist town",                  # 163
    "[E] freelance event photographer",                                   # 164
    "[E] landscape business owner with a small crew",                    # 165
    "[E] mobile phone repair shop owner",                                # 166
    # ── C — Conventional (167-199) ───────────────────────────────────────────
    "[C] mid-level civil servant in local government planning",          # 167
    "[C] medical billing specialist at a hospital group",                # 168
    "[C] court reporter transitioning to freelance captioning",          # 169
    "[C] tax preparer who volunteers for low-income families",           # 170
    "[C] building code inspector",                                        # 171
    "[C] election administrator in a divided district",                  # 172
    "[C] air traffic controller dealing with shift fatigue",             # 173
    "[C] postal carrier who walks the same route daily",                 # 174
    "[C] legal aid paralegal managing case files",                       # 175
    "[C] payroll administrator at a mid-size company",                   # 176
    "[C] compliance officer at a pharmaceutical company",                # 177
    "[C] records manager at a county courthouse",                        # 178
    "[C] quality assurance tester at a software company",                # 179
    "[C] bank teller studying for a finance degree at night",            # 180
    "[C] dental office manager",                                          # 181
    "[C] inventory specialist at a distribution warehouse",              # 182
    "[C] technical writer for open-source projects",                     # 183
    "[C] school district administrative coordinator",                    # 184
    "[C] veterinary clinic office manager",                              # 185
    "[C] social media coordinator at a human-rights nonprofit",          # 186
    "[C] library assistant cataloguing a special collection",            # 187
    "[C] insurance underwriter reviewing commercial policies",           # 188
    "[C] city bus dispatcher coordinating routes",                       # 189
    "[C] pharmacy technician at a busy hospital",                        # 190
    "[C] corporate travel coordinator",                                   # 191
    "[C] data entry specialist at a research institute",                 # 192
    "[C] loan processor at a community bank",                            # 193
    "[C] human resources assistant at a manufacturing firm",             # 194
    "[C] medical records technician at a teaching hospital",             # 195
    "[C] purchasing clerk for a school district",                        # 196
    "[C] cemetery groundskeeper who writes local history",               # 197
    "[C] Montessori school administrator with a neuroscience background", # 198
    "[C] funeral director who also runs grief support groups",           # 199
]

# Pool of 200 diverse names spanning many cultures.
# Sampled without replacement at runtime — guarantees no duplicates across any run.
_NAME_POOL = [
    # English / American
    "James Carter", "Emma Sullivan", "Noah Williams", "Olivia Bennett",
    "Liam Harrison", "Ava Mitchell", "Mason Turner", "Sophia Evans",
    "Ethan Brooks", "Isabella Clark", "Logan Reed", "Charlotte Hughes",
    "Caleb Foster", "Amelia Parker", "Jackson Cole", "Harper Stone",
    "Owen Murphy", "Abigail Ross", "Elijah Ward", "Emily Nichols",
    "Lucas Hayes", "Grace Stewart", "Henry Griffin", "Lily Crawford",
    "Samuel Perry", "Zoe Russell", "Benjamin Powell", "Chloe Barnes",
    "Daniel Jenkins", "Natalie Coleman",
    # East Asian
    "Wei Zhang", "Mei Lin Chen", "Hiro Tanaka", "Yuki Nakamura",
    "Jae-won Kim", "Soo-yeon Park", "Xiao Ming Liu", "Fang Yu Wang",
    "Kenji Ishida", "Akemi Fujimoto", "Min-jun Lee", "Ji-yeon Choi",
    "Bao Nguyen", "Lan Thi Pham", "Zhen Wei Li", "Ruixue Huang",
    "Daiki Suzuki", "Haruki Watanabe", "Soyeon Han", "Guang Wei Zhao",
    # South Asian
    "Priya Sharma", "Arjun Patel", "Meera Krishnan", "Rahul Gupta",
    "Ananya Iyer", "Vikram Singh", "Sanjana Nair", "Aditya Kumar",
    "Neha Joshi", "Rohan Mehta", "Kavya Reddy", "Aarav Malhotra",
    "Ishaan Verma", "Pooja Bhatt", "Divya Pillai", "Kiran Rao",
    "Nikhil Saxena", "Aditi Chandra", "Farhan Hussain", "Nadia Ahmed",
    # African / Nigerian
    "Chioma Okafor", "Emeka Eze", "Amara Diallo", "Kofi Asante",
    "Fatima Bello", "Adaeze Nwosu", "Seun Adeyemi", "Zainab Musa",
    "Oluwaseun Afolabi", "Ngozi Ike", "Taiwo Olawale", "Chiamaka Okonkwo",
    "Kwame Mensah", "Yetunde Adesanya", "Tunde Bakare", "Amina Suleiman",
    "Obi Egwu", "Blessing Uchenna", "Nnamdi Okoye", "Chidinma Agu",
    # Hispanic / Latino
    "Sofia Ramirez", "Carlos Herrera", "Isabella Flores", "Miguel Torres",
    "Valentina Cruz", "Diego Morales", "Camila Vega", "Mateo Jimenez",
    "Lucia Reyes", "Andres Castillo", "Gabriela Fuentes", "Alejandro Mendoza",
    "Mariana Rivas", "Sebastian Vargas", "Paula Navarro", "Rafael Gutierrez",
    "Elena Perez", "Joaquin Salazar", "Natalia Rojas", "Fernando Delgado",
    # Middle Eastern / Arab
    "Layla Hassan", "Omar Khalil", "Fatima Al-Rashid", "Yousef Mansour",
    "Nour Abboud", "Tariq Farouk", "Hana Nasser", "Karim Aziz",
    "Rania Haddad", "Samir Qureshi", "Yasmin Saleh", "Amir Masoud",
    "Dina Barakat", "Faris Taleb", "Leila Karimi",
    # Eastern European
    "Aleksei Volkov", "Natasha Ivanova", "Marek Nowak", "Katarzyna Kowalska",
    "Pavel Novotny", "Vera Sokolova", "Bogdan Popescu", "Dmitri Morozov",
    "Oksana Kovalenko", "Radovan Jovic", "Maja Horvat", "Szymon Wieczorek",
    "Agnieszka Duda", "Yuri Borisov", "Irena Blazevic",
    # Scandinavian / Nordic
    "Erik Lindqvist", "Astrid Sorensen", "Magnus Eriksson", "Ingrid Andersen",
    "Lars Johansson", "Freya Hansen", "Bjorn Gustafsson", "Sigrid Larsson",
    "Leif Christiansen", "Solveig Nilsson", "Tor Magnusson", "Hanna Berg",
    "Finn Olsen", "Maja Petersen", "Sven Holm",
    # Southeast Asian
    "Thanh Nguyen", "Siti Rahayu", "Budi Santoso", "Malee Thongchai",
    "Rico Dela Cruz", "Anh Thi Hoang", "Wiriya Suksawat", "Reza Hartono",
    "Crisanto Reyes", "Kyaw Zin Win", "Malai Chaiyasit", "Dang Van Minh",
    "Putri Wulandari", "Sakda Bunyarit", "Nithya Subramaniam",
    # West / Central African
    "Moussa Coulibaly", "Aissatou Diallo", "Seydou Traore", "Mariam Toure",
    "Cheikh Ndiaye", "Khady Sow", "Ibrahima Keita", "Fanta Camara",
    "Lamine Diop", "Rokhaya Ba",
    # Iranian / Persian
    "Dariush Tehrani", "Nasrin Shirazi", "Bahram Rostami", "Shirin Moradi",
    "Cyrus Alizadeh", "Maryam Hosseini", "Reza Ghorbani", "Parisa Kazemi",
    "Kaveh Sadeghi", "Zahra Rahimi",
    # Turkish / Greek
    "Mehmet Yilmaz", "Ayse Kaya", "Nikolaos Papadopoulos", "Eleni Dimitriou",
    "Baris Ozturk", "Zeynep Demir", "Kostas Georgiou", "Maria Papadaki",
    "Hasan Celik", "Despina Stavros",
]

_HINT_PERSONALITY = [
    # 16PF factor poles: A=Warmth, B=Reasoning, C=Emotional Stability,
    # E=Dominance, F=Liveliness, G=Rule-Consciousness, H=Social Boldness,
    # I=Sensitivity, L=Vigilance, M=Abstractedness, N=Privateness,
    # O=Apprehension, Q1=Openness to Change, Q2=Self-Reliance,
    # Q3=Perfectionism, Q4=Tension
    # ── R — Realistic (0-33) ─────────────────────────────────────────────────
    ["high Emotional Stability (C)", "low Abstractedness (M)", "high Self-Reliance (Q2)"],      # 0
    ["high Emotional Stability (C)", "low Sensitivity (I)", "high Perfectionism (Q3)"],         # 1
    ["low Abstractedness (M)", "high Dominance (E)", "high Emotional Stability (C)"],           # 2
    ["high Self-Reliance (Q2)", "low Abstractedness (M)", "high Rule-Consciousness (G)"],       # 3
    ["low Tension (Q4)", "high Sensitivity (I)", "low Abstractedness (M)"],                     # 4
    ["high Perfectionism (Q3)", "high Self-Reliance (Q2)", "low Liveliness (F)"],               # 5
    ["low Abstractedness (M)", "high Rule-Consciousness (G)", "high Emotional Stability (C)"],  # 6
    ["low Liveliness (F)", "high Privateness (N)", "high Self-Reliance (Q2)"],                  # 7
    ["low Tension (Q4)", "high Warmth (A)", "low Abstractedness (M)"],                          # 8
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "low Apprehension (O)"],          # 9
    ["high Self-Reliance (Q2)", "high Vigilance (L)", "low Abstractedness (M)"],                # 10
    ["high Perfectionism (Q3)", "high Dominance (E)", "high Tension (Q4)"],                     # 11
    ["high Dominance (E)", "high Social Boldness (H)", "low Sensitivity (I)"],                  # 12
    ["high Emotional Stability (C)", "low Abstractedness (M)", "high Rule-Consciousness (G)"],  # 13
    ["low Abstractedness (M)", "high Emotional Stability (C)", "high Openness to Change (Q1)"], # 14
    ["high Self-Reliance (Q2)", "low Liveliness (F)", "high Emotional Stability (C)"],          # 15
    ["high Warmth (A)", "low Tension (Q4)", "high Openness to Change (Q1)"],                    # 16
    ["high Perfectionism (Q3)", "high Self-Reliance (Q2)", "low Liveliness (F)"],               # 17
    ["high Sensitivity (I)", "high Openness to Change (Q1)", "low Abstractedness (M)"],         # 18
    ["low Abstractedness (M)", "high Emotional Stability (C)", "low Liveliness (F)"],           # 19
    ["high Dominance (E)", "high Rule-Consciousness (G)", "low Sensitivity (I)"],               # 20
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "high Warmth (A)"],               # 21
    ["high Dominance (E)", "low Abstractedness (M)", "high Emotional Stability (C)"],           # 22
    ["high Emotional Stability (C)", "high Privateness (N)", "low Tension (Q4)"],               # 23
    ["high Liveliness (F)", "low Abstractedness (M)", "high Tension (Q4)"],                     # 24
    ["high Perfectionism (Q3)", "low Liveliness (F)", "high Privateness (N)"],                  # 25
    ["high Warmth (A)", "low Abstractedness (M)", "high Rule-Consciousness (G)"],               # 26
    ["high Emotional Stability (C)", "high Warmth (A)", "low Abstractedness (M)"],              # 27
    ["high Emotional Stability (C)", "high Self-Reliance (Q2)", "low Apprehension (O)"],        # 28
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "high Privateness (N)"],          # 29
    ["high Emotional Stability (C)", "high Dominance (E)", "low Apprehension (O)"],             # 30
    ["high Perfectionism (Q3)", "high Sensitivity (I)", "low Abstractedness (M)"],              # 31
    ["high Emotional Stability (C)", "low Abstractedness (M)", "high Privateness (N)"],         # 32
    ["high Rule-Consciousness (G)", "high Self-Reliance (Q2)", "low Abstractedness (M)"],       # 33
    # ── I — Investigative (34-66) ────────────────────────────────────────────
    ["high Reasoning (B)", "high Openness to Change (Q1)", "low Liveliness (F)"],               # 34
    ["high Reasoning (B)", "high Abstractedness (M)", "low Dominance (E)"],                     # 35
    ["high Abstractedness (M)", "high Self-Reliance (Q2)", "high Tension (Q4)"],                # 36
    ["high Reasoning (B)", "high Sensitivity (I)", "high Openness to Change (Q1)"],             # 37
    ["high Reasoning (B)", "high Liveliness (F)", "high Self-Reliance (Q2)"],                   # 38
    ["high Apprehension (O)", "high Reasoning (B)", "low Social Boldness (H)"],                 # 39
    ["high Perfectionism (Q3)", "high Reasoning (B)", "high Privateness (N)"],                  # 40
    ["high Openness to Change (Q1)", "high Reasoning (B)", "high Apprehension (O)"],            # 41
    ["high Reasoning (B)", "high Tension (Q4)", "low Liveliness (F)"],                          # 42
    ["high Abstractedness (M)", "high Social Boldness (H)", "high Reasoning (B)"],              # 43
    ["high Sensitivity (I)", "high Self-Reliance (Q2)", "high Reasoning (B)"],                  # 44
    ["high Reasoning (B)", "low Abstractedness (M)", "high Emotional Stability (C)"],           # 45
    ["high Abstractedness (M)", "high Openness to Change (Q1)", "high Self-Reliance (Q2)"],     # 46
    ["high Vigilance (L)", "high Reasoning (B)", "high Privateness (N)"],                       # 47
    ["high Abstractedness (M)", "high Reasoning (B)", "high Openness to Change (Q1)"],          # 48
    ["high Perfectionism (Q3)", "high Reasoning (B)", "low Social Boldness (H)"],               # 49
    ["high Reasoning (B)", "high Tension (Q4)", "high Self-Reliance (Q2)"],                     # 50
    ["high Sensitivity (I)", "high Abstractedness (M)", "high Reasoning (B)"],                  # 51
    ["high Reasoning (B)", "low Abstractedness (M)", "high Liveliness (F)"],                    # 52
    ["high Rule-Consciousness (G)", "high Reasoning (B)", "low Openness to Change (Q1)"],       # 53
    ["high Vigilance (L)", "high Reasoning (B)", "high Perfectionism (Q3)"],                    # 54
    ["high Rule-Consciousness (G)", "high Warmth (A)", "high Reasoning (B)"],                   # 55
    ["high Self-Reliance (Q2)", "high Privateness (N)", "high Reasoning (B)"],                  # 56
    ["high Emotional Stability (C)", "high Reasoning (B)", "low Abstractedness (M)"],           # 57
    ["high Warmth (A)", "high Reasoning (B)", "high Self-Reliance (Q2)"],                       # 58
    ["high Abstractedness (M)", "high Reasoning (B)", "low Liveliness (F)"],                    # 59
    ["high Rule-Consciousness (G)", "high Reasoning (B)", "high Privateness (N)"],              # 60
    ["high Openness to Change (Q1)", "high Reasoning (B)", "low Tension (Q4)"],                 # 61
    ["high Warmth (A)", "high Abstractedness (M)", "high Reasoning (B)"],                       # 62
    ["high Reasoning (B)", "low Abstractedness (M)", "high Openness to Change (Q1)"],           # 63
    ["low Liveliness (F)", "high Reasoning (B)", "high Perfectionism (Q3)"],                    # 64
    ["high Emotional Stability (C)", "high Self-Reliance (Q2)", "high Reasoning (B)"],          # 65
    ["high Reasoning (B)", "high Openness to Change (Q1)", "high Perfectionism (Q3)"],          # 66
    # ── A — Artistic (67-99) ─────────────────────────────────────────────────
    ["high Sensitivity (I)", "high Openness to Change (Q1)", "high Tension (Q4)"],              # 67
    ["high Social Boldness (H)", "high Vigilance (L)", "high Openness to Change (Q1)"],         # 68
    ["high Sensitivity (I)", "high Abstractedness (M)", "high Openness to Change (Q1)"],        # 69
    ["high Openness to Change (Q1)", "high Abstractedness (M)", "low Liveliness (F)"],          # 70
    ["high Warmth (A)", "high Sensitivity (I)", "high Openness to Change (Q1)"],                # 71
    ["high Self-Reliance (Q2)", "high Openness to Change (Q1)", "high Sensitivity (I)"],        # 72
    ["high Dominance (E)", "high Sensitivity (I)", "high Liveliness (F)"],                      # 73
    ["high Abstractedness (M)", "high Self-Reliance (Q2)", "high Sensitivity (I)"],             # 74
    ["high Sensitivity (I)", "low Tension (Q4)", "high Openness to Change (Q1)"],               # 75
    ["high Openness to Change (Q1)", "high Dominance (E)", "high Sensitivity (I)"],             # 76
    ["high Emotional Stability (C)", "low Abstractedness (M)", "high Openness to Change (Q1)"], # 77
    ["high Sensitivity (I)", "high Privateness (N)", "high Openness to Change (Q1)"],           # 78
    ["high Vigilance (L)", "high Openness to Change (Q1)", "high Liveliness (F)"],              # 79
    ["high Sensitivity (I)", "high Self-Reliance (Q2)", "low Rule-Consciousness (G)"],          # 80
    ["high Perfectionism (Q3)", "high Sensitivity (I)", "low Liveliness (F)"],                  # 81
    ["high Openness to Change (Q1)", "high Liveliness (F)", "high Sensitivity (I)"],            # 82
    ["high Abstractedness (M)", "high Openness to Change (Q1)", "high Self-Reliance (Q2)"],     # 83
    ["high Privateness (N)", "high Abstractedness (M)", "high Sensitivity (I)"],                # 84
    ["high Sensitivity (I)", "high Abstractedness (M)", "high Apprehension (O)"],               # 85
    ["high Warmth (A)", "high Sensitivity (I)", "high Tension (Q4)"],                           # 86
    ["high Liveliness (F)", "high Sensitivity (I)", "high Social Boldness (H)"],                # 87
    ["high Perfectionism (Q3)", "high Sensitivity (I)", "low Liveliness (F)"],                  # 88
    ["high Abstractedness (M)", "high Self-Reliance (Q2)", "high Openness to Change (Q1)"],     # 89
    ["high Sensitivity (I)", "high Social Boldness (H)", "high Openness to Change (Q1)"],       # 90
    ["high Perfectionism (Q3)", "high Abstractedness (M)", "low Social Boldness (H)"],          # 91
    ["high Liveliness (F)", "high Sensitivity (I)", "low Rule-Consciousness (G)"],              # 92
    ["high Reasoning (B)", "high Openness to Change (Q1)", "high Sensitivity (I)"],             # 93
    ["high Social Boldness (H)", "high Liveliness (F)", "high Openness to Change (Q1)"],        # 94
    ["high Sensitivity (I)", "high Perfectionism (Q3)", "low Social Boldness (H)"],             # 95
    ["high Social Boldness (H)", "high Sensitivity (I)", "high Self-Reliance (Q2)"],            # 96
    ["high Warmth (A)", "high Liveliness (F)", "high Sensitivity (I)"],                         # 97
    ["high Dominance (E)", "high Sensitivity (I)", "high Perfectionism (Q3)"],                  # 98
    ["high Warmth (A)", "high Sensitivity (I)", "low Tension (Q4)"],                            # 99
    # ── S — Social (100-133) ─────────────────────────────────────────────────
    ["high Emotional Stability (C)", "high Warmth (A)", "low Privateness (N)"],                 # 100
    ["high Warmth (A)", "high Apprehension (O)", "low Dominance (E)"],                          # 101
    ["high Dominance (E)", "high Warmth (A)", "high Tension (Q4)"],                             # 102
    ["high Warmth (A)", "high Sensitivity (I)", "high Apprehension (O)"],                       # 103
    ["high Warmth (A)", "high Sensitivity (I)", "high Tension (Q4)"],                           # 104
    ["high Warmth (A)", "high Emotional Stability (C)", "high Sensitivity (I)"],                # 105
    ["high Social Boldness (H)", "high Warmth (A)", "high Dominance (E)"],                      # 106
    ["high Warmth (A)", "high Apprehension (O)", "high Emotional Stability (C)"],               # 107
    ["high Emotional Stability (C)", "high Warmth (A)", "low Tension (Q4)"],                    # 108
    ["high Warmth (A)", "high Sensitivity (I)", "low Dominance (E)"],                           # 109
    ["high Warmth (A)", "high Sensitivity (I)", "high Apprehension (O)"],                       # 110
    ["high Warmth (A)", "high Openness to Change (Q1)", "low Vigilance (L)"],                   # 111
    ["high Warmth (A)", "high Dominance (E)", "high Tension (Q4)"],                             # 112
    ["high Warmth (A)", "high Emotional Stability (C)", "high Openness to Change (Q1)"],        # 113
    ["high Warmth (A)", "high Openness to Change (Q1)", "high Apprehension (O)"],               # 114
    ["high Warmth (A)", "high Liveliness (F)", "high Social Boldness (H)"],                     # 115
    ["high Warmth (A)", "high Emotional Stability (C)", "high Dominance (E)"],                  # 116
    ["high Dominance (E)", "high Warmth (A)", "high Social Boldness (H)"],                      # 117
    ["high Warmth (A)", "high Perfectionism (Q3)", "low Abstractedness (M)"],                   # 118
    ["high Warmth (A)", "low Abstractedness (M)", "high Liveliness (F)"],                       # 119
    ["high Warmth (A)", "high Rule-Consciousness (G)", "high Tension (Q4)"],                    # 120
    ["high Warmth (A)", "high Liveliness (F)", "high Sensitivity (I)"],                         # 121
    ["high Warmth (A)", "high Openness to Change (Q1)", "high Social Boldness (H)"],            # 122
    ["high Warmth (A)", "high Emotional Stability (C)", "low Abstractedness (M)"],              # 123
    ["high Warmth (A)", "high Sensitivity (I)", "high Self-Reliance (Q2)"],                     # 124
    ["high Warmth (A)", "high Sensitivity (I)", "high Privateness (N)"],                        # 125
    ["high Warmth (A)", "high Openness to Change (Q1)", "high Abstractedness (M)"],             # 126
    ["high Warmth (A)", "low Tension (Q4)", "high Sensitivity (I)"],                            # 127
    ["high Dominance (E)", "high Social Boldness (H)", "high Warmth (A)"],                      # 128
    ["high Social Boldness (H)", "high Warmth (A)", "high Emotional Stability (C)"],            # 129
    ["high Warmth (A)", "high Emotional Stability (C)", "low Vigilance (L)"],                   # 130
    ["high Warmth (A)", "high Sensitivity (I)", "low Abstractedness (M)"],                      # 131
    ["high Perfectionism (Q3)", "high Warmth (A)", "high Emotional Stability (C)"],             # 132
    ["high Warmth (A)", "high Sensitivity (I)", "high Emotional Stability (C)"],                # 133
    # ── E — Enterprising (134-166) ───────────────────────────────────────────
    ["high Dominance (E)", "high Social Boldness (H)", "high Tension (Q4)"],                    # 134
    ["high Social Boldness (H)", "high Reasoning (B)", "high Dominance (E)"],                   # 135
    ["high Perfectionism (Q3)", "high Dominance (E)", "low Liveliness (F)"],                    # 136
    ["high Social Boldness (H)", "high Liveliness (F)", "high Dominance (E)"],                  # 137
    ["high Dominance (E)", "high Reasoning (B)", "low Sensitivity (I)"],                        # 138
    ["high Warmth (A)", "high Dominance (E)", "high Sensitivity (I)"],                          # 139
    ["high Dominance (E)", "high Reasoning (B)", "high Social Boldness (H)"],                   # 140
    ["high Social Boldness (H)", "high Dominance (E)", "high Vigilance (L)"],                   # 141
    ["high Emotional Stability (C)", "low Abstractedness (M)", "high Dominance (E)"],           # 142
    ["high Liveliness (F)", "high Social Boldness (H)", "high Dominance (E)"],                  # 143
    ["high Emotional Stability (C)", "high Dominance (E)", "low Openness to Change (Q1)"],      # 144
    ["high Openness to Change (Q1)", "high Dominance (E)", "high Perfectionism (Q3)"],          # 145
    ["high Social Boldness (H)", "high Liveliness (F)", "high Tension (Q4)"],                   # 146
    ["high Liveliness (F)", "high Openness to Change (Q1)", "high Dominance (E)"],              # 147
    ["high Reasoning (B)", "high Vigilance (L)", "high Dominance (E)"],                         # 148
    ["high Warmth (A)", "high Sensitivity (I)", "high Dominance (E)"],                          # 149
    ["high Emotional Stability (C)", "high Vigilance (L)", "low Liveliness (F)"],               # 150
    ["high Warmth (A)", "high Dominance (E)", "low Tension (Q4)"],                              # 151
    ["high Social Boldness (H)", "high Dominance (E)", "low Apprehension (O)"],                 # 152
    ["high Dominance (E)", "high Vigilance (L)", "high Tension (Q4)"],                          # 153
    ["high Rule-Consciousness (G)", "high Dominance (E)", "high Tension (Q4)"],                 # 154
    ["high Liveliness (F)", "high Social Boldness (H)", "high Perfectionism (Q3)"],             # 155
    ["high Social Boldness (H)", "high Dominance (E)", "high Warmth (A)"],                      # 156
    ["high Liveliness (F)", "high Social Boldness (H)", "high Dominance (E)"],                  # 157
    ["high Warmth (A)", "high Social Boldness (H)", "high Openness to Change (Q1)"],            # 158
    ["high Emotional Stability (C)", "high Dominance (E)", "low Abstractedness (M)"],           # 159
    ["high Liveliness (F)", "high Social Boldness (H)", "low Tension (Q4)"],                    # 160
    ["high Dominance (E)", "high Social Boldness (H)", "high Liveliness (F)"],                  # 161
    ["high Warmth (A)", "high Dominance (E)", "low Vigilance (L)"],                             # 162
    ["high Warmth (A)", "high Liveliness (F)", "high Dominance (E)"],                           # 163
    ["high Social Boldness (H)", "high Openness to Change (Q1)", "high Liveliness (F)"],        # 164
    ["high Dominance (E)", "high Emotional Stability (C)", "low Abstractedness (M)"],           # 165
    ["high Self-Reliance (Q2)", "high Dominance (E)", "low Tension (Q4)"],                      # 166
    # ── C — Conventional (167-199) ───────────────────────────────────────────
    ["high Rule-Consciousness (G)", "high Perfectionism (Q3)", "low Abstractedness (M)"],       # 167
    ["high Perfectionism (Q3)", "high Rule-Consciousness (G)", "low Liveliness (F)"],           # 168
    ["high Perfectionism (Q3)", "low Abstractedness (M)", "high Emotional Stability (C)"],      # 169
    ["high Warmth (A)", "high Rule-Consciousness (G)", "low Tension (Q4)"],                     # 170
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "high Vigilance (L)"],            # 171
    ["high Rule-Consciousness (G)", "high Emotional Stability (C)", "high Perfectionism (Q3)"], # 172
    ["high Emotional Stability (C)", "high Perfectionism (Q3)", "high Tension (Q4)"],           # 173
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "high Warmth (A)"],               # 174
    ["high Perfectionism (Q3)", "high Rule-Consciousness (G)", "high Warmth (A)"],              # 175
    ["high Perfectionism (Q3)", "low Abstractedness (M)", "low Liveliness (F)"],                # 176
    ["high Rule-Consciousness (G)", "high Vigilance (L)", "high Perfectionism (Q3)"],           # 177
    ["high Perfectionism (Q3)", "high Privateness (N)", "low Liveliness (F)"],                  # 178
    ["high Perfectionism (Q3)", "high Reasoning (B)", "high Openness to Change (Q1)"],          # 179
    ["high Rule-Consciousness (G)", "high Apprehension (O)", "high Tension (Q4)"],              # 180
    ["high Warmth (A)", "high Perfectionism (Q3)", "low Abstractedness (M)"],                   # 181
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "high Perfectionism (Q3)"],       # 182
    ["high Reasoning (B)", "high Perfectionism (Q3)", "high Self-Reliance (Q2)"],               # 183
    ["high Rule-Consciousness (G)", "high Warmth (A)", "high Perfectionism (Q3)"],              # 184
    ["high Warmth (A)", "high Rule-Consciousness (G)", "high Perfectionism (Q3)"],              # 185
    ["high Openness to Change (Q1)", "high Rule-Consciousness (G)", "high Liveliness (F)"],     # 186
    ["high Perfectionism (Q3)", "low Liveliness (F)", "high Sensitivity (I)"],                  # 187
    ["high Rule-Consciousness (G)", "high Perfectionism (Q3)", "low Openness to Change (Q1)"],  # 188
    ["high Emotional Stability (C)", "high Rule-Consciousness (G)", "low Abstractedness (M)"],  # 189
    ["high Rule-Consciousness (G)", "high Perfectionism (Q3)", "low Tension (Q4)"],             # 190
    ["high Perfectionism (Q3)", "high Rule-Consciousness (G)", "low Abstractedness (M)"],       # 191
    ["high Perfectionism (Q3)", "low Abstractedness (M)", "high Privateness (N)"],              # 192
    ["high Rule-Consciousness (G)", "high Perfectionism (Q3)", "high Warmth (A)"],              # 193
    ["high Warmth (A)", "high Rule-Consciousness (G)", "low Abstractedness (M)"],               # 194
    ["high Perfectionism (Q3)", "high Rule-Consciousness (G)", "high Privateness (N)"],         # 195
    ["high Rule-Consciousness (G)", "low Abstractedness (M)", "high Perfectionism (Q3)"],       # 196
    ["high Self-Reliance (Q2)", "low Abstractedness (M)", "high Privateness (N)"],              # 197
    ["high Openness to Change (Q1)", "high Warmth (A)", "high Reasoning (B)"],                  # 198
    ["high Warmth (A)", "high Emotional Stability (C)", "high Rule-Consciousness (G)"],         # 199
]


def compute_tech_affinity(persona: PersonaCard) -> float:
    """Estimate tech-savviness from age, occupation, and hobbies.

    Returns a float in [0.05, 1.0].
    """
    # Age curve: linear from 0.9 at 18 to 0.2 at 80
    age = max(18, min(80, persona.age))
    score = 0.9 - (age - 18) * (0.7 / 62)

    occ = persona.occupation.lower()
    # Boost for tech-adjacent occupations
    tech_keywords = ("tech", "software", "data", "developer", "programmer",
                     "engineer", "devops", "it ", "analyst")
    if any(kw in occ for kw in tech_keywords):
        score += 0.15

    # Penalty for manual-labor occupations
    manual_keywords = ("plumber", "farmer", "electrician", "carpenter",
                       "mechanic", "welder", "mason", "janitor")
    if any(kw in occ for kw in manual_keywords):
        score -= 0.1

    # Boost for tech-adjacent hobbies
    hobby_text = " ".join(h.lower() for h in persona.hobbies)
    if any(kw in hobby_text for kw in ("gaming", "coding", "programming",
                                        "keyboard", "video game", "speedrun")):
        score += 0.1

    return round(max(0.05, min(1.0, score)), 2)


def build_public_information(persona: PersonaCard) -> Dict[str, Dict[str, Any]]:
    """Build the tiered public_information dict from a PersonaCard.

    Five tiers, from most to least disclosure:

      intimate    — everything (close friends, family)
      close       — most things (good friends, close colleagues)
      active      — basics (regular contacts, work acquaintances)
      acquaintance — name + job only (people you see around)
      stranger     — name only (no shared history)

    This dict is stored on the PersonaCard and consumed by
    social_knowledge_injector to seed each agent's initial knowledge of peers.
    """
    location = persona.demographics.get("location", "")
    traits = ", ".join(persona.personality_traits) if persona.personality_traits else ""
    sleep = (
        f"sleeps {int(persona.sleep_start_hour):02d}:00"
        f"–{int(persona.sleep_end_hour):02d}:00"
    )

    return {
        "intimate": {
            "name": persona.name,
            "age": persona.age,
            "occupation": persona.occupation,
            "location": location,
            "personality": traits,
            "hobbies": list(persona.hobbies),
            "values": list(persona.values),
            "current_concerns": list(persona.current_concerns),
            "backstory": persona.backstory,
            "speaking_style": persona.speaking_style,
            "sleep_schedule": sleep,
            "work_schedule": persona.work_schedule,
            "daily_routine": persona.daily_routine_notes,
            "relationships": dict(persona.relationships),
        },
        "close": {
            "name": persona.name,
            "age": persona.age,
            "occupation": persona.occupation,
            "location": location,
            "personality": traits,
            "hobbies": list(persona.hobbies[:2]),
            "values": list(persona.values[:2]),
        },
        "active": {
            "name": persona.name,
            "occupation": persona.occupation,
            "location": location,
            "communication_style": persona.communication_style,
        },
        "acquaintance": {
            "name": persona.name,
            "occupation": persona.occupation,
        },
        "stranger": {
            "name": persona.name,
        },
    }


def personas_to_slugs(personas: List[PersonaCard]) -> List[str]:
    """Convert persona names to unique lowercase slug IDs.

    e.g. "Dr. Maya Johal" -> "maya_johal"
    Appends a numeric suffix if duplicates arise.
    """
    seen: Dict[str, int] = {}
    slugs: List[str] = []
    for p in personas:
        raw = p.name.strip()
        raw = re.sub(r'^(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s*', '', raw)
        slug = re.sub(r'[^a-z0-9]+', '_', raw.lower()).strip('_')
        if not slug:
            slug = "persona"
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}_{seen[slug]}"
        else:
            seen[slug] = 0
        slugs.append(slug)
    return slugs


def _profile_to_skeleton(profile: Dict[str, Any]) -> str:
    """Convert a HINT_PROFILES entry into a readable bullet-point skeleton."""
    lines = []
    lines.append(f"- Age: {profile['age']}")
    lines.append(f"- Gender: {profile['gender']}")
    lines.append(f"- Marital status: {profile['marital']}")
    lines.append(f"- Race/ethnicity: {profile['race']}")
    lines.append(f"- Income level: {profile['income']}")
    lines.append(f"- Nationality: {profile['nat']}")
    lines.append(f"- Mother language: {profile['lang']}")
    lines.append(f"- Education: {profile['edu']}")
    lines.append(f"- Zodiac: {profile['zodiac']}")
    lines.append(f"- Idol: {profile['idol']}")
    lines.append(f"- Motto: \"{profile['motto']}\"")
    lines.append(f"- Preferred platform: {profile['platform']}")
    lines.append(f"- Background: {profile['bg']}")
    lines.append(f"- Speaking style: {profile['speak']}")
    lines.append(f"- Hobbies: {', '.join(profile['hobbies'])}")
    lines.append(f"- Values: {', '.join(profile['values'])}")
    lines.append(f"- Current concerns: {', '.join(profile['concerns'])}")
    people = profile.get("people", {})
    if people:
        rel_parts = [f"{name} ({rel})" for name, rel in people.items()]
        lines.append(f"- Key people: {', '.join(rel_parts)}")
    lines.append(f"- Favourite movie: {profile.get('movie', 'N/A')}")
    lines.append(f"- Sport: {profile.get('sport', 'N/A')}")
    sleep = profile.get("sleep", [23.0, 7.0, "9-5 weekdays"])
    lines.append(f"- Sleep: {sleep[0]}–{sleep[1]}, work schedule: {sleep[2]}")
    return "\n".join(lines)


class PersonaFactory:
    """Generate rich persona cards for simulation agents."""

    def __init__(self, llm_client: Any, seed: int = 42):
        self.llm_client = llm_client
        self.seed = seed

    def generate_personas(
        self,
        n_agents: int,
        layer_distribution: Optional[Dict[int, int]] = None,
        dry_run: bool = False,
    ) -> List[PersonaCard]:
        """Generate n_agents diverse, richly-detailed persona cards.

        Names are pre-assigned by sampling from _NAME_POOL without replacement,
        so duplicates are impossible regardless of LLM output.
        """
        if layer_distribution is None:
            layer_distribution = self._default_layer_distribution(n_agents)

        # Pre-sample unique names for all agents
        rng = np.random.default_rng(self.seed)
        if n_agents <= len(_NAME_POOL):
            assigned_names = list(rng.choice(_NAME_POOL, size=n_agents, replace=False))
        else:
            # More agents than pool entries — cycle with index suffix
            base = list(rng.choice(_NAME_POOL, size=len(_NAME_POOL), replace=False))
            assigned_names = []
            for i in range(n_agents):
                name = base[i % len(base)]
                assigned_names.append(name if i < len(base) else f"{name} {i // len(base) + 2}")

        tasks: List[Dict[str, str]] = []
        layer_assignments: List[int] = []
        hint_idx = 0

        for layer, count in sorted(layer_distribution.items()):
            for _ in range(count):
                hint = DIVERSITY_HINTS[hint_idx % len(DIVERSITY_HINTS)]
                profile = HINT_PROFILES[hint_idx % len(HINT_PROFILES)]
                hint_idx += 1
                skeleton = _profile_to_skeleton(profile)
                prompt = PERSONA_USER_TEMPLATE.format(
                    diversity_hint=hint, profile_skeleton=skeleton,
                )
                tasks.append({"system": PERSONA_SYSTEM_PROMPT, "user": prompt, "max_tokens": 8192, "tags": {"phase": "persona_gen"}})
                layer_assignments.append(layer)

        if dry_run:
            return self._generate_placeholder_personas(n_agents, layer_assignments, assigned_names)

        log.info("Generating %d rich personas via LLM...", n_agents)
        responses = self.llm_client.generate_batch(tasks)

        personas = []
        for i, (resp, layer) in enumerate(zip(responses, layer_assignments)):
            profile = HINT_PROFILES[i % len(HINT_PROFILES)]
            persona = self._parse_persona(resp, layer, index=i, hint_profile=profile)
            persona.name = assigned_names[i]   # override LLM name with pool name
            persona.public_information = build_public_information(persona)
            personas.append(persona)

        log.info("Generated %d personas", len(personas))
        return personas

    def _parse_persona(
        self, response: str, layer: int, index: int,
        hint_profile: Optional[Dict[str, Any]] = None,
    ) -> PersonaCard:
        """Parse LLM response into a PersonaCard, tolerating partial JSON.

        If *hint_profile* is provided, its demographic/fun fields are merged
        into the demographics dict so zodiac, idol, motto, etc. are always
        present even when the LLM omits them.
        """
        try:
            data = _repair_json(response)

            demographics = data.get("demographics", {})
            # Merge HINT_PROFILES fields into demographics
            if hint_profile:
                _profile_demo = {
                    "marital_status": hint_profile.get("marital", ""),
                    "race": hint_profile.get("race", ""),
                    "income": hint_profile.get("income", ""),
                    "nationality": hint_profile.get("nat", ""),
                    "mother_language": hint_profile.get("lang", ""),
                    "zodiac": hint_profile.get("zodiac", ""),
                    "idol": hint_profile.get("idol", ""),
                    "motto": hint_profile.get("motto", ""),
                    "social_platform": hint_profile.get("platform", ""),
                    "favourite_movie": hint_profile.get("movie", ""),
                    "favourite_sport": hint_profile.get("sport", ""),
                }
                for k, v in _profile_demo.items():
                    if v and k not in demographics:
                        demographics[k] = v

            p = PersonaCard(
                name=data.get("name", f"Agent_{index:04d}"),
                age=int(data.get("age", 30)),
                occupation=data.get("occupation", "unknown"),
                demographics=demographics,
                personality_traits=data.get("personality_traits", []),
                expertise=data.get("expertise", []),
                communication_style=data.get("communication_style", "neutral"),
                education_level=data.get("education_level", ""),
                speaking_style=data.get("speaking_style", ""),
                backstory=data.get("backstory", ""),
                hobbies=data.get("hobbies", []),
                current_concerns=data.get("current_concerns", []),
                values=data.get("values", []),
                relationships=data.get("relationships", {}),
                dunbar_layer=layer,
                sleep_start_hour=float(data.get("sleep_start_hour", 23.0)),
                sleep_end_hour=float(data.get("sleep_end_hour", 7.0)),
                work_schedule=data.get("work_schedule", "9-5 weekdays"),
                daily_routine_notes=data.get("daily_routine_notes", ""),
            )
            p.tech_affinity = compute_tech_affinity(p)
            p.public_information = build_public_information(p)
            return p
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            log.warning("Failed to parse persona %d: %s", index, e)
            fallback = PersonaCard(name=f"Agent_{index:04d}", dunbar_layer=layer)
            fallback.tech_affinity = compute_tech_affinity(fallback)
            return fallback

    def _default_layer_distribution(self, n: int) -> Dict[int, int]:
        """Distribute agents across Dunbar layers proportionally.

        For small runs we still spread across layers to get variety in how
        much social knowledge agents share about each other.
        """
        if n <= 2:
            return {1: n}
        if n <= 10:
            # Spread evenly across layers 1, 2, 3 for small runs
            n1 = max(1, n // 3)
            n2 = max(1, n // 3)
            n3 = n - n1 - n2
            dist = {1: n1, 2: n2}
            if n3 > 0:
                dist[3] = n3
            return dist
        # For larger runs: proportional to Dunbar layer sizes
        ratios = {1: 5, 2: 15, 3: 50, 4: 130}
        total_ratio = sum(ratios.values())
        dist = {}
        assigned = 0
        for layer in [1, 2, 3]:
            count = max(1, round(n * ratios[layer] / total_ratio))
            dist[layer] = count
            assigned += count
        dist[4] = max(1, n - assigned)
        return dist

    def _generate_placeholder_personas(
        self, n: int, layer_assignments: List[int], assigned_names: List[str],
    ) -> List[PersonaCard]:
        """Generate placeholder personas for dry-run mode.

        Each agent gets a distinct character drawn from HINT_PROFILES (demographics,
        background, speaking style, hobbies, etc.), _HINT_PERSONALITY (16PF traits),
        and DIVERSITY_HINTS (Holland-coded vocation), with names taken from the
        pre-sampled assigned_names list.
        """
        personas = []
        for i in range(n):
            idx = i % len(DIVERSITY_HINTS)
            layer = layer_assignments[i] if i < len(layer_assignments) else 4
            name = assigned_names[i]
            hint = DIVERSITY_HINTS[idx]
            prof = HINT_PROFILES[idx]
            traits = _HINT_PERSONALITY[idx]

            # Extract sleep data from profile
            sleep_data = prof["sleep"]
            sleep_start, sleep_end, work_sched = sleep_data[0], sleep_data[1], sleep_data[2]

            # Strip Holland prefix "[X] " from hint for occupation
            occupation = hint[4:] if hint.startswith("[") and len(hint) > 4 else hint

            # Build demographics dict with all profile fields
            demographics = {
                "gender": prof["gender"],
                "marital_status": prof["marital"],
                "race": prof["race"],
                "income": prof["income"],
                "nationality": prof["nat"],
                "mother_language": prof["lang"],
                "zodiac": prof["zodiac"],
                "idol": prof["idol"],
                "motto": prof["motto"],
                "social_platform": prof["platform"],
                "favourite_movie": prof["movie"],
                "favourite_sport": prof["sport"],
            }

            p = PersonaCard(
                name=name,
                age=prof["age"],
                occupation=occupation,
                demographics=demographics,
                personality_traits=traits,
                expertise=[],
                communication_style="casual",
                education_level=prof["edu"],
                sleep_start_hour=sleep_start,
                sleep_end_hour=sleep_end,
                work_schedule=work_sched,
                daily_routine_notes="",
                speaking_style=prof["speak"],
                backstory=prof["bg"],
                hobbies=list(prof["hobbies"]),
                current_concerns=list(prof["concerns"]),
                values=list(prof["values"]),
                relationships=dict(prof["people"]),
                dunbar_layer=layer,
            )
            p.tech_affinity = compute_tech_affinity(p)
            p.public_information = build_public_information(p)
            personas.append(p)
        return personas
