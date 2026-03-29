# server/task_generator.py
"""
Procedural policy scenario generator.
Generates variants of each task by swapping domains, severity, and actor types.
Activated by calling generate_task_variants() and updating TASK_REGISTRY.
"""
from __future__ import annotations
import random
from typing import Iterator, Dict

DOMAIN_VARIANTS = {
    "social_media": ["gaming_platform", "professional_network", "dating_app", "news_forum"],
    "corporate_hr": ["startup_culture", "remote_first_company", "government_agency", "university"],
    "ecommerce": ["marketplace", "subscription_service", "auction_platform", "b2b_procurement"],
}

ACTOR_VARIANTS = ["new_user", "power_user", "verified_creator", "anonymous_account", "enterprise_client"]


def generate_easy_variant(base_task: Dict, domain: str) -> Dict:
    variant = dict(base_task)
    variant["task_id"] = f"task_easy_{domain.replace(' ', '_')}"
    variant["description"] = base_task["description"].replace("social media platform", domain)
    return variant


def generate_task_variants(base_task: Dict, n: int = 5) -> Iterator[Dict]:
    domain_list = list(DOMAIN_VARIANTS.values())
    all_domains = [d for sublist in domain_list for d in sublist]
    for i in range(min(n, len(all_domains))):
        domain = all_domains[i]
        yield generate_easy_variant(base_task, domain)
