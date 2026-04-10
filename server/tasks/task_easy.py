# server/tasks/task_easy.py
EASY_TASK = {
    "task_id": "task_easy",
    "difficulty": "easy",
    "description": (
        "Modern workplace policies for AI, remote work, and gig workers are often vague. "
        "Identify ONE genuinely ambiguous term (e.g. 'appropriate', 'substantial', 'reasonable') "
        "and propose a specific, measurable definition to ensure consistent governance."
    ),
    "data_corpus": [
        # AI Use (10)
        {"id": "ai_001", "type": "AI_use", "content": "Employee used Claude to draft a README without disclosure", "system_action": "pending"},
        {"id": "ai_002", "type": "AI_use", "content": "Dev used Copilot for 90% of a feature branch", "system_action": "pending"},
        {"id": "ai_003", "type": "AI_use", "content": "Marketing used Midjourney for ad assets", "system_action": "pending"},
        {"id": "ai_004", "type": "AI_use", "content": "HR used an AI filter to reject 500 resumes", "system_action": "pending"},
        {"id": "ai_005", "type": "AI_use", "content": "Sales used a deepfake voice for a cold call test", "system_action": "pending"},
        {"id": "ai_006", "type": "AI_use", "content": "Legal used ChatGPT to summarize a contract", "system_action": "pending"},
        {"id": "ai_007", "type": "AI_use", "content": "Intern submitted AI-generated report as original research", "system_action": "pending"},
        {"id": "ai_008", "type": "AI_use", "content": "Support agent used AI to translate customer tickets", "system_action": "pending"},
        {"id": "ai_009", "type": "AI_use", "content": "Data scientist used LLM to generate synthetic training data", "system_action": "pending"},
        {"id": "ai_010", "type": "AI_use", "content": "UX designer used AI to generate 50 user personas", "system_action": "pending"},
        
        # Remote Work (10)
        {"id": "remote_001", "type": "remote_work", "content": "Employee worked from a public park using insecure Wi-Fi", "system_action": "pending"},
        {"id": "remote_002", "type": "remote_work", "content": "Manager requested 'always-on' webcam for remote staff", "system_action": "pending"},
        {"id": "remote_003", "type": "remote_work", "content": "Employee moved to Bali for 6 months without notifying HR", "system_action": "pending"},
        {"id": "remote_004", "type": "remote_work", "content": "Staff member taking 4-hour midday breaks but working until midnight", "system_action": "pending"},
        {"id": "remote_005", "type": "remote_work", "content": "Remote dev sharing a workspace with a competitor's employee", "system_action": "pending"},
        {"id": "remote_006", "type": "remote_work", "content": "Employee claiming home office expenses for a luxury yacht", "system_action": "pending"},
        {"id": "remote_007", "type": "remote_work", "content": "Video call background showing sensitive prototype designs", "system_action": "pending"},
        {"id": "remote_008", "type": "remote_work", "content": "Employee missed 3 consecutive standups due to 'bad signal'", "system_action": "pending"},
        {"id": "remote_009", "type": "remote_work", "content": "Staffer using a mouse-jiggler to appear active on Slack", "system_action": "pending"},
        {"id": "remote_010", "type": "remote_work", "content": "Employee using company laptop for a side-hustle during hours", "system_action": "pending"},
        
        # Gig Worker (10)
        {"id": "gig_001", "type": "gig_worker", "content": "Freelancer accessed internal Slack without a signed NDA", "system_action": "pending"},
        {"id": "gig_002", "type": "gig_worker", "content": "Contractor working for three direct competitors simultaneously", "system_action": "pending"},
        {"id": "gig_003", "type": "gig_worker", "content": "Temp worker sharing proprietary API keys on a public forum", "system_action": "pending"},
        {"id": "gig_004", "type": "gig_worker", "content": "Gig designer using company account for personal project storage", "system_action": "pending"},
        {"id": "gig_005", "type": "gig_worker", "content": "Contractor requested health benefits after 12 months of 40h/week", "system_action": "pending"},
        {"id": "gig_006", "type": "gig_worker", "content": "Freelancer sub-contracted their work to a third party without consent", "system_action": "pending"},
        {"id": "gig_007", "type": "gig_worker", "content": "Gig coder refused to use company's version control system", "system_action": "pending"},
        {"id": "gig_008", "type": "gig_worker", "content": "Contractor accessed sensitive HR server for 'formatting ideas'", "system_action": "pending"},
        {"id": "gig_009", "type": "gig_worker", "content": "Temp staff member wearing competitor's merch in office", "system_action": "pending"},
        {"id": "gig_010", "type": "gig_worker", "content": "Freelancer claimed 80 hours of work for 20 actual hours", "system_action": "pending"},
        # Red Herrings (Noise for Staff-Level filtering)
        {"id": "noise_001", "type": "staff_social", "content": "Employee asked on Slack if anyone wants to order pizza", "system_action": "pending"},
        {"id": "noise_002", "type": "office_infra", "content": "The coffee machine in the 3rd floor breakroom is leaking", "system_action": "pending"},
        {"id": "noise_003", "type": "social_event", "content": "Reminder: The annual company picnic is next Friday at 2 PM", "system_action": "pending"},
        {"id": "noise_004", "type": "it_notice", "content": "Scheduled maintenance on the internal portal this Sunday at 1 AM", "system_action": "pending"},
    ],
    "current_policies": [
        {"id": "pol_wplace_001", "text": "Employees must use AI tools in an appropriate and ethical manner."},
        {"id": "pol_wplace_002", "text": "Remote work environments must be reasonable and professional."},
        {"id": "pol_wplace_003", "text": "Gig workers should maintain a respectful relationship with firm intellectual property."},
        {"id": "pol_wplace_004", "text": "Substantial use of external automation requires management approval."},
        {"id": "pol_noise_999", "text": "The company mascot 'OpenBot' shall always be depicted wearing a blue tie in internal slides."}, # Noise Policy
    ],
    "known_ambiguous_terms": ["appropriate", "ethical", "reasonable", "professional", "respectful", "substantial"],
    "red_herrings": ["pizza", "coffee machine", "picnic", "mascot", "blue tie", "lunch", "weather"],
    "num_policies": 5,
    "num_data_points": 34,
}
