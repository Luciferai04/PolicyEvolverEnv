# server/tasks/task_medium.py
MEDIUM_TASK = {
    "task_id": "task_medium",
    "difficulty": "medium",
    "description": (
        "A corporate HR policy set is missing rules for several emerging workplace "
        "scenarios involving AI tools, remote work, and gig workers. "
        "Identify ONE genuine policy gap and propose a specific new rule to address it."
    ),
    "data_corpus": [
        # AI Use (15)
        {"id": "med_ai_001", "type": "AI_use", "content": "Employee used ChatGPT to write client proposal without disclosure", "system_action": "pending"},
        {"id": "med_ai_002", "type": "AI_use", "content": "Manager used AI to generate performance review for employee", "system_action": "pending"},
        {"id": "med_ai_003", "type": "AI_use", "content": "Employee submitted AI-written code as original work", "system_action": "pending"},
        {"id": "med_ai_004", "type": "AI_use", "content": "HR used AI tool to screen resumes — bias concerns raised", "system_action": "pending"},
        {"id": "med_ai_005", "type": "AI_use", "content": "AI generated a deceptive internal memo appearing to come from CEO", "system_action": "pending"},
        {"id": "med_ai_006", "type": "AI_use", "content": "Marketing team used AI to create a deepfake campaign video", "system_action": "pending"},
        {"id": "med_ai_007", "type": "AI_use", "content": "Product team fed proprietary roadmap into public LLM for strategy", "system_action": "pending"},
        {"id": "med_ai_008", "type": "AI_use", "content": "Software dev using local AI to leak source code through patterns", "system_action": "pending"},
        {"id": "med_ai_009", "type": "AI_use", "content": "Customer support using AI without human-in-the-loop oversight", "system_action": "pending"},
        {"id": "med_ai_010", "type": "AI_use", "content": "Employee using AI to bypass mandatory security training modules", "system_action": "pending"},
        {"id": "med_ai_011", "type": "AI_use", "content": "AI model suggesting illegal tax evasion strategies to finance team", "system_action": "pending"},
        {"id": "med_ai_012", "type": "AI_use", "content": "Researcher using AI to forge peer-review comments", "system_action": "pending"},
        {"id": "med_ai_013", "type": "AI_use", "content": "Automated recruitment bot discriminating based on zip code", "system_action": "pending"},
        {"id": "med_ai_014", "type": "AI_use", "content": "Employee using unauthorized GPT-4 API on production servers", "system_action": "pending"},
        {"id": "med_ai_015", "type": "AI_use", "content": "Executive assistant using AI to falsify meeting minutes", "system_action": "pending"},

        # Remote Work (15)
        {"id": "med_rem_001", "type": "remote_work", "content": "Employee attended video call from a coffee shop, client data visible", "system_action": "pending"},
        {"id": "med_rem_002", "type": "remote_work", "content": "Employee shared screen showing salary data on public WiFi", "system_action": "pending"},
        {"id": "med_rem_003", "type": "remote_work", "content": "Employee worked from another country for 3 months without HR approval", "system_action": "pending"},
        {"id": "med_rem_004", "type": "remote_work", "content": "Remote dev using a proxy to hide their actual location from security", "system_action": "pending"},
        {"id": "med_rem_005", "type": "remote_work", "content": "Employee working two full-time jobs at once via clever calendar management", "system_action": "pending"},
        {"id": "med_rem_006", "type": "remote_work", "content": "Staffer using personal smart-speaker which listens to confidential calls", "system_action": "pending"},
        {"id": "med_rem_007", "type": "remote_work", "content": "Employee working from a shared Airbnb with non-employees present", "system_action": "pending"},
        {"id": "med_rem_008", "type": "remote_work", "content": "Regional manager demanding 10pm status updates from cross-timezone staff", "system_action": "pending"},
        {"id": "med_rem_009", "type": "remote_work", "content": "Insecure home IoT device used as bridge for corporate ransomware", "system_action": "pending"},
        {"id": "med_rem_010", "type": "remote_work", "content": "Employee refusing to return company assets after transitioning to full remote", "system_action": "pending"},
        {"id": "med_rem_011", "type": "remote_work", "content": "Manager tracking keystrokes without prior remote-policy disclosure", "system_action": "pending"},
        {"id": "med_rem_012", "type": "remote_work", "content": "Staff member leaking sensitive info via shared home printer cache", "system_action": "pending"},
        {"id": "med_rem_013", "type": "remote_work", "content": "Employee moving to states with higher tax burdens without informing company", "system_action": "pending"},
        {"id": "med_rem_014", "type": "remote_work", "content": "Remote team using unapproved messaging apps for sensitive IP discussion", "system_action": "pending"},
        {"id": "med_rem_015", "type": "remote_work", "content": "Staffing claims home office status but working from a tropical villa", "system_action": "pending"},

        # Gig Worker (15)
        {"id": "med_gig_001", "type": "gig_worker", "content": "Contractor accessed proprietary codebase after project ended", "system_action": "pending"},
        {"id": "med_gig_002", "type": "gig_worker", "content": "Freelancer posted client project on portfolio without permission", "system_action": "pending"},
        {"id": "med_gig_003", "type": "gig_worker", "content": "Contractor working for a direct competitor simultaneously", "system_action": "pending"},
        {"id": "med_gig_004", "type": "gig_worker", "content": "Gig coder requesting access to payroll database for 'insight'", "system_action": "pending"},
        {"id": "med_gig_005", "type": "gig_worker", "content": "Freelance designer using copyrighted stock without licensing for client", "system_action": "pending"},
        {"id": "med_gig_006", "type": "gig_worker", "content": "Contractor threatening to delete code repo over minor billing delay", "system_action": "pending"},
        {"id": "med_gig_007", "type": "gig_worker", "content": "Gig-platform worker using client credentials for a different startup", "system_action": "pending"},
        {"id": "med_gig_008", "type": "gig_worker", "content": "Freelancer claiming to be a 50-person agency but is one person with AI", "system_action": "pending"},
        {"id": "med_gig_009", "type": "gig_worker", "content": "Contractor accessing server logs to extract user email lists", "system_action": "pending"},
        {"id": "med_gig_010", "type": "gig_worker", "content": "Gig worker suing for tenure benefits after 'permanently temporary' status", "system_action": "pending"},
        {"id": "med_gig_011", "type": "gig_worker", "content": "Temporary admin sharing executive travel schedules with rivals", "system_action": "pending"},
        {"id": "med_gig_012", "type": "gig_worker", "content": "Consultant refusing to hand over documentation until 'exit bonus' paid", "system_action": "pending"},
        {"id": "med_gig_013", "type": "gig_worker", "content": "Agency worker using client API key for a personal web-scraping bot", "system_action": "pending"},
        {"id": "med_gig_014", "type": "gig_worker", "content": "Contractor claiming patent rights on code written for the firm", "system_action": "pending"},
        {"id": "med_gig_015", "type": "gig_worker", "content": "Freelance writer using AI to generate 1,000 keyword-stuffed articles", "system_action": "pending"},

        # Edge Cases (5)
        {"id": "med_edge_001", "type": "cross_border_tax", "content": "Software team distributed across 12 countries with no tax nexus set", "system_action": "pending"},
        {"id": "med_edge_002", "type": "mental_health", "content": "Employee burnout linked to 24/7 Slack culture in remote team", "system_action": "pending"},
        {"id": "med_edge_003", "type": "security", "content": "Employee using a corporate laptop for high-risk crypto-mining", "system_action": "pending"},
        {"id": "med_edge_004", "type": "data_sovereignty", "content": "EU client data stored on a server in a region without adequacy", "system_action": "pending"},
        {"id": "med_edge_005", "type": "ethics", "content": "AI system used to predict which employees are likely to quit", "system_action": "pending"},
    ],
    "current_policies": [
        {"id": "pol_hr_001", "text": "Employees must maintain confidentiality of client information at all times."},
        {"id": "pol_hr_002", "text": "All employees are expected to comply with the company code of conduct."},
        {"id": "pol_hr_003", "text": "Contractors must sign an NDA before beginning any project."},
        {"id": "pol_hr_004", "text": "Employees working remotely must have a secure, dedicated workspace."},
        {"id": "pol_hr_005", "text": "Any intellectual property created during employment belongs to the company."},
    ],
    "uncovered_domains": ["AI_use", "gig_worker_post_engagement", "cross_border_remote", "mental_health_governance"],
    "num_policies": 5,
    "num_data_points": 50,
}
